#include "KNN_GPU.cuh"

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#include "cuda_header.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct PearsonIntermediate {
    // from all viewers who rated both movie X and Y.
    float x; // sum of ratings for movie X
    float y; // sum of ratings for movie Y
    float xy; // sum of product of ratings for movies X and Y
    float xx; // sum of square of ratings for movie X
    float yy; // sum of square of ratings for movie Y
    int cnt; // number of viewers who rated both movies
};

// a struct for one entry in the training data
struct RatingTriple {
    int m;
    int u;
    int r;
};

// a struct for holding integer ranges, used in pointer arithmetic
struct Range {
    int start;
    int end;
};

__global__
void cuda_euclidean_kernel(const unsigned int *gpu_movie_data, const int n_movies,
                           const int n_features, float *gpu_movie_distances) {

    // compute the current thread index
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    // while the thread corresponds to a valid entry in the distance matrix
    while (thread_index < n_movies * n_movies) {

        // compute indices of movie pair
        int i = thread_index / n_movies;
        int j = thread_index % n_movies;

        if (i == j) {
            // movie is distance -1.0 from itself
            gpu_movie_distances[n_movies * i + j] = -1.0;

        }
        else {

            // calculate distance
            float dist = 0.0;
            for(int k = 0; k < n_features; k++) {

                float x_ik = (float) gpu_movie_data[i * n_features + k];
                float x_jk = (float) gpu_movie_data[j * n_features + k];

                dist += pow(x_ik - x_jk, 2);
            }

            gpu_movie_distances[n_movies * i + j] = dist; // pow(dist, 0.5);

        }

        // advance thread index
        thread_index += blockDim.x * gridDim.x;
    }

}

__global__
void cuda_pearson_intermediate_kernel(const RatingTriple *gpu_movie_ratings,
                                      const Range *gpu_user_ranges,
                                      const RatingTriple *gpu_user_ratings,
                                      const int n_ratings,
                                      const int n_movies_split,
                                      const int n_movies,
                                      PearsonIntermediate* gpu_intermediates) {
    
    // compute the current thread index
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    // while the thread corresponds to a valid rating entry
    while (thread_index < n_ratings) {

        // obtain movie and user indices for this rating
        int mi = gpu_movie_ratings[thread_index].m;
        int u  = gpu_movie_ratings[thread_index].u;
        int ri = gpu_movie_ratings[thread_index].r;
        
        // accumulate Pearson intermediates over all movies mj rated by user u
        int start = (gpu_user_ranges[u]).start;
        int end   = (gpu_user_ranges[u]).end;
        for (int j = start; j < end; j++) {
            int mj = gpu_user_ratings[j].m;
            int rj = gpu_user_ratings[j].r;
            
            PearsonIntermediate* e = gpu_intermediates + (mi * n_movies + mj);
            
            atomicAdd(&(e->x),  (float) ri);
            atomicAdd(&(e->y),  (float) rj);
            atomicAdd(&(e->xy), (float) ri * rj);
            atomicAdd(&(e->xx), (float) ri * ri);
            atomicAdd(&(e->yy), (float) rj * rj);
            atomicAdd(&(e->cnt), 1);
        }
        
        // advance thread index
        thread_index += blockDim.x * gridDim.x;
    }
}

__global__
void cuda_pearson_kernel(PearsonIntermediate* gpu_intermediates,
                         const int n_movies_split,
                         const int n_movies,
                         const int movie_low,
                         float *gpu_movie_distances) {
    
    // compute the current thread index
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    // while the thread corresponds to a valid entry in the gpu_intermediates matrix
    while (thread_index < n_movies_split * n_movies) {

        // compute indices of movie, movie pair
        int i = thread_index / n_movies;
        int j = thread_index % n_movies;
        
        // unpack Pearson intermediate for this movie, movie pair
        PearsonIntermediate e = gpu_intermediates[i * n_movies + j];
        float x = e.x;
        float y = e.y;
        float xy = e.xy;
        float xx = e.xx;
        float yy = e.yy;
        int cnt = e.cnt;
        
        // process Pearson intermediate to obtain Pearson correlation coef.
        if (i + movie_low == j) {
            gpu_movie_distances[i * n_movies + j] = 1.0;
        } 
        else {
            if (cnt == 0)
            {
                gpu_movie_distances[i * n_movies + j] = 0;
            }
            else
            {
                float result = (cnt * xy - x * y) / (sqrt(cnt * xx - x*x) * sqrt(cnt * yy - y*y));

                // Set to 0 if result in NaN
                // https://stackoverflow.com/questions/570669/checking-if-a-double-or-float-is-nan-in-c
                gpu_movie_distances[i * n_movies + j] = (result != result) ? 0.0 : result;
                
            }
        }

        // advance thread index
        thread_index += blockDim.x * gridDim.x;
    }
    
}

__global__
void cuda_prediction_kernel(const float *gpu_movie_distances,
                            const Range *gpu_user_ranges,
                            RatingTriple *gpu_user_ratings,
                            const int k,
                            const int n_movies,
                            const int pred_movies,
                            const int pred_users,
                            float *gpu_knn_predictions) {
                                
    // compute the current thread index
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    // while the thread corresponds to a valid query pair
    while (thread_index < pred_movies * pred_users) {

        // compute indices of movie, user pair
        int u = thread_index / pred_movies;
        int m = thread_index % pred_movies;
        
        int start = gpu_user_ranges[u].start;
        int end   = gpu_user_ranges[u].end;
        
        // if there are insufficient movies rated by this user, use default
        if (end - start < k) {
            gpu_knn_predictions[thread_index] = 0.0;
        }
        // otherwise, computed weighted average of top k movies rated by this user
        else {
            
            float *topk_weights = new float[k];
            
            // copy array before sorting in place
            RatingTriple* topk = new RatingTriple[end - start];
            for (int i = 0; i < end - start; i++) {
                topk[i] = gpu_user_ratings[start + i];
            }
            
            // find top k, using selection sort (in place)
            for (int i = 0; i < k; i++) {
                
                // we assume all coefficients are between -1 and 1, and larger = more similar
                int j_max = i;
                float max_coef = -1.0;
                
                // identify largest unsorted element
                for (int j = i; j < end - start; j++) {
                    
                    int m2 = topk[j].m;
                    float coef = gpu_movie_distances[m * n_movies + m2];
                    if (coef > max_coef) {
                        max_coef = coef;
                        j_max = j;
                    }
                    
                }
                
                // swap largest unsorted element with ith element
                RatingTriple tmp = topk[i];
                topk[i] = topk[j_max];
                topk[j_max] = tmp;
                
                topk_weights[i] = max_coef;
            }
            
            // computed weighted average of top k
            float weight_sum = 0.0;
            float rating_sum = 0.0;
            for (int i = 0; i < k; i++)
            {
                rating_sum += topk_weights[i] * ((float) topk[i].r);
                weight_sum += topk_weights[i];
            }
            
            gpu_knn_predictions[thread_index] = 
                (weight_sum == 0.0) ? 0.0 : rating_sum / weight_sum;
            
            delete[] topk_weights;
            delete[] topk;
        }

        // advance thread index
        thread_index += blockDim.x * gridDim.x;
    }
    
}

int cuda_call_euclidean_kernel(const unsigned int blocks,
                               const unsigned int threads_per_block,
                               const unsigned int *movie_data,
                               const int n_movies,
                               const int n_features,
                               float *movie_distances) {

    // allocate and copy data to gpu memory
    unsigned int* gpu_movie_data;
    gpuErrchk( cudaMalloc((void **) &gpu_movie_data, n_movies * n_features * sizeof(unsigned int)) );
    cudaMemcpy(gpu_movie_data, movie_data, n_movies * n_features * sizeof(unsigned int), cudaMemcpyHostToDevice);

    float* gpu_movie_distances;
    gpuErrchk( cudaMalloc((void **) &gpu_movie_distances, n_movies * n_movies * sizeof(float)) );

    // call kernel
    cuda_euclidean_kernel<<<blocks, threads_per_block>>>(gpu_movie_data, n_movies, n_features, gpu_movie_distances);

    // check for errors on kernel call
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 0;
    }
    else {
        fprintf(stderr, "No kernel error detected\n");
    }

    // copy output to cpu memory
    cudaMemcpy(movie_distances, gpu_movie_distances, n_movies * n_movies * sizeof(float), cudaMemcpyDeviceToHost);

    // free gpu memory
    cudaFree(gpu_movie_data);
    cudaFree(gpu_movie_distances);

    return 1; // return success flag (for now)
}

int cuda_call_pearson_kernel(const unsigned int blocks,
                             const unsigned int threads_per_block,
                             std::vector<std::unordered_map<int, int>> &movie_ratings,
                             std::vector<std::unordered_map<int, int>> &user_ratings,
                             const int n_movies,
                             const int n_users,
                             float *movie_distances) {
    
    // convert unordered_maps to arrays to be loaded onto GPU
    
    // first, calculate total number of entries in split
    int n_ratings = 0;
    for (int u = 0; u < n_users; u++)
        n_ratings += user_ratings[u].size();
    
    // all ratings, grouped by user index
    RatingTriple* user_ratings_arr = new RatingTriple[n_ratings];
    // range of indices in user_ratings_arr corresponding to each user
    Range* user_ranges = new Range[n_users];
    int i = 0;
    
    for (int u = 0; u < n_users; u++) {
        
        user_ranges[u].start = i;
        
        // loop through unordered map for this user
        for (auto mr: user_ratings[u]) {
            
            int m = mr.first;
            int r = mr.second;
            
            user_ratings_arr[i] = {m, u, r};
            i += 1;
        }
        
        user_ranges[u].end = i;
    }
    
    // allocate user arrays
    RatingTriple* gpu_user_ratings;
    gpuErrchk( cudaMalloc((void **) &gpu_user_ratings, n_ratings * sizeof(RatingTriple)) );
    cudaMemcpy(gpu_user_ratings, user_ratings_arr, n_ratings * sizeof(RatingTriple), cudaMemcpyHostToDevice);
    
    Range* gpu_user_ranges;
    gpuErrchk( cudaMalloc((void **) &gpu_user_ranges, n_users * sizeof(Range)) );
    cudaMemcpy(gpu_user_ranges, user_ranges, n_users * sizeof(Range), cudaMemcpyHostToDevice);
    
    // in case n_movies is too large, split calculation into multiple runs to
    // stay within GPU memory limits
    for (int split = 0; 1000 * split < n_movies; split++) {
        
        // which rows in movie distances are calculated in this split
        int movie_low = 1000 * split;
        int movie_high = min(1000 * (split + 1), n_movies);
        int n_movies_split = movie_high - movie_low;
        
        // convert unordered_maps to arrays to be loaded onto GPU
        
        // first, calculate number of entries in split
        int n_ratings_split = 0;
        for (int m = movie_low; m < movie_high; m++)
            n_ratings_split += movie_ratings[m].size();
        
        // all ratings, grouped by movie index
        RatingTriple* movie_ratings_arr = new RatingTriple[n_ratings_split];
        i = 0;
        
        for (int m = movie_low; m < movie_high; m++) {
            
            // loop through unordered map for this movie
            for (auto ur: movie_ratings[m]) {
                
                int u = ur.first;
                int r = ur.second;
                
                movie_ratings_arr[i] = {m - movie_low, u, r};
                i += 1;
            }
        }
        
        // allocate and copy data to gpu memory
        RatingTriple* gpu_movie_ratings;
        gpuErrchk( cudaMalloc((void **) &gpu_movie_ratings, n_ratings_split * sizeof(RatingTriple)) );
        cudaMemcpy(gpu_movie_ratings, movie_ratings_arr, n_ratings_split * sizeof(RatingTriple), cudaMemcpyHostToDevice);
        
        PearsonIntermediate* gpu_intermediates;
        gpuErrchk( cudaMalloc((void **) &gpu_intermediates, n_movies_split * n_movies * sizeof(PearsonIntermediate)) );
        
        float* gpu_movie_distances;
        gpuErrchk( cudaMalloc((void **) &gpu_movie_distances, n_movies_split * n_movies * sizeof(float)) );
        
        // call intermediates kernel
        cuda_pearson_intermediate_kernel<<<blocks, threads_per_block>>>(
                                          gpu_movie_ratings,
                                          gpu_user_ranges,
                                          gpu_user_ratings,
                                          n_ratings_split,
                                          n_movies_split,
                                          n_movies,
                                          gpu_intermediates);
                                          
        // check for errors on kernel call
        cudaError err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
            return 0;
        }
        
        // call Pearson coefficient kernel
        cuda_pearson_kernel<<<blocks, threads_per_block>>>(gpu_intermediates,
                                                           n_movies_split,
                                                           n_movies,
                                                           movie_low,
                                                           gpu_movie_distances);
                                                           
        // check for errors on kernel call
        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
            return 0;
        }
        
        // copy output to cpu memory
        cudaMemcpy(movie_distances + movie_low * n_movies, gpu_movie_distances, n_movies_split * n_movies * sizeof(float), cudaMemcpyDeviceToHost);
        
        // free cpu memory
        delete[] movie_ratings_arr;
        
        // free gpu memory
        cudaFree(gpu_intermediates);
        cudaFree(gpu_movie_distances);
        cudaFree(gpu_movie_ratings);
    }
    
    // free cpu memory
    delete[] user_ratings_arr;
    delete[] user_ranges;
    
    // free gpu memory
    cudaFree(gpu_user_ranges);
    cudaFree(gpu_user_ratings);

    return 1; // return success flag (for now)
    
}

int cuda_call_prediction_kernel(const unsigned int blocks,
                                const unsigned int threads_per_block,
                                std::vector<std::unordered_map<int, int>> &user_ratings,
                                const float *movie_distances, 
                                const int k,
                                const int n_movies,
                                const int n_users,
                                float *knn_predictions,
                                const int pred_movies,
                                const int pred_users,
                                const bool pearson) {
                                    
    // convert distances to similarity coefficients
    float * movie_distances_2 = new float [n_movies * n_movies];
    for (int i = 0; i < n_movies * n_movies; i++) {
        movie_distances_2[i] = pearson ? movie_distances[i] : 1.0 / (1.0 + movie_distances[i]);
    }
    
    // convert unordered map to array to load onto GPU
    
    // first, calculate total number of entries
    int n_ratings = 0;
    for (int u = 0; u < n_users; u++)
        n_ratings += user_ratings[u].size();
    
    // all ratings, grouped by user index
    RatingTriple* user_ratings_arr = new RatingTriple[n_ratings];
    // range of indices in user_ratings_arr corresponding to each user
    Range* user_ranges = new Range[n_users];
    int i = 0;
    
    for (int u = 0; u < n_users; u++) {
        
        user_ranges[u].start = i;
        
        // loop through unordered map for this user
        for (auto mr: user_ratings[u]) {
            
            int m = mr.first;
            int r = mr.second;
            
            user_ratings_arr[i] = {m, u, r};
            i += 1;
        }
        
        user_ranges[u].end = i;
    }
    
    // allocate and copy data to gpu memory
    float* gpu_movie_distances;
    gpuErrchk( cudaMalloc((void **) &gpu_movie_distances, n_movies * n_movies * sizeof(float)) );
    cudaMemcpy(gpu_movie_distances, movie_distances_2, n_movies * n_movies * sizeof(float), cudaMemcpyHostToDevice);
    
    RatingTriple* gpu_user_ratings;
    gpuErrchk( cudaMalloc((void **) &gpu_user_ratings, n_ratings * sizeof(RatingTriple)) );
    cudaMemcpy(gpu_user_ratings, user_ratings_arr, n_ratings * sizeof(RatingTriple), cudaMemcpyHostToDevice);
    
    Range* gpu_user_ranges;
    gpuErrchk( cudaMalloc((void **) &gpu_user_ranges, n_users * sizeof(Range)) );
    cudaMemcpy(gpu_user_ranges, user_ranges, n_users * sizeof(Range), cudaMemcpyHostToDevice);
    
    float* gpu_knn_predictions;
    gpuErrchk( cudaMalloc((void **) &gpu_knn_predictions, pred_movies * pred_users * sizeof(float)) );
    
    // call kernel
    cuda_prediction_kernel<<<blocks, threads_per_block>>>(gpu_movie_distances,
                                                          gpu_user_ranges,
                                                          gpu_user_ratings,
                                                          k,
                                                          n_movies,
                                                          pred_movies,
                                                          pred_users,
                                                          gpu_knn_predictions);
    
    // check for errors on kernel call
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 0;
    }
    
    // copy output to cpu memory
    cudaMemcpy(knn_predictions, gpu_knn_predictions, pred_movies * pred_users * sizeof(float), cudaMemcpyDeviceToHost);
    
    // free cpu memory
    delete[] user_ratings_arr;
    delete[] user_ranges;
    delete[] movie_distances_2;
    
    // free gpu memory
    cudaFree(gpu_user_ratings);
    cudaFree(gpu_user_ranges);
    cudaFree(gpu_knn_predictions);

    return 1; // return success flag (for now)
}


