/**
 * GPU-Accelerated KNN
 */

#include <cassert>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string>
#include <stdio.h>
#include <bits/stdc++.h>
#include <iostream>
#include <cuda_runtime.h>
#include "KNN.h"
#include "MovieLensParser.h"
#include "KNN_GPU.cuh"


/*
* Helper utils to print the movie distance and rating matrices
*/
void print_array(std::string n, unsigned int ** uint_data, int xdim, int ydim)
{
    std::cout << "Starting " << n;
    for (int i = 0; i < xdim; i++)
    {
        std::cout << std::endl;
        for (int j = 0; j < ydim; j++)
        {
            std::cout << (*uint_data)[i * ydim + j] << ", ";
        }
    }
    std::cout << std::endl << "Loaded " << n << std::endl;
}

void print_array(std::string n, float ** float_data, int xdim, int ydim)
{
    std::cout << "Starting " << n;
    for (int i = 0; i < xdim; i++)
    {
        std::cout << std::endl;
        for (int j = 0; j < ydim; j++)
        {
            std::cout << (*float_data)[i * ydim + j] << ", ";
        }
    }
    std::cout << std::endl << "Loaded " << n << std::endl;
}

void print_ratings(int &n_movies, std::vector<std::unordered_map<int, int>> &movie_ratings)
{
    std::cout << "Printing Complete Movie Ratings" << std::endl;
    for (int i = 0; i < n_movies; i++)
    {
        for (auto it : movie_ratings[i])
        {
            std::cout << "Movie " << i + 1 << " was rated by user "
                << it.first << " with a rating of " << it.second << std::endl;
        }
    }
    std::cout << "Finished Printing All User-Movie Ratings" << std::endl;
}

// Check to make sure two arrays are the same
void compare_arrays(float *cpu_array, float *gpu_array, int n, float epsilon = 0.01)
{
    bool correct = true;
    float cpu_val;
    float gpu_val;
    for (int i = 0; i < n; i++)
    {
        cpu_val = cpu_array[i];
        gpu_val = gpu_array[i];

        if (abs(cpu_val - gpu_val) > epsilon) {
            correct = false;
            std::cout << "CPU value of " << cpu_val <<
                " is different from GPU value of " << gpu_val << 
                " at index " << i << std::endl;
        }
    }
    assert(correct);
}

void print_random_entries(float *cpu_array, float *gpu_array, int n_print, int n_max)
{
    printf("Printing %d of %d entries\n", n_print, n_max);
    printf("i\tcpu\tgpu\n");
    
    for (int i = 0; i < n_print; i++) {
        int j = rand() % n_max;
        printf("%d\t%.3f\t%.3f\n", j, cpu_array[j], gpu_array[j]);
    }
}

/*
KNN Calculations with Pearson Correlation Metric
*/
void pearson_movie_lens(bool is100k, int gpu, int k_val, bool verbose)
{
    // MovieLens Dataset defaults
    std::string rating_name = "ml-20m/ratings.csv";
    if (is100k) {
        rating_name = "ml-100k/u.data";
    }

    int n_movies = 27278;
    int n_users = 138493;
    std::string dataset_size = "20 million";
    if (is100k) {
        n_movies = 1682;
        n_users = 943;
        dataset_size = "100 thousand";
    }

    clock_t t;
    t = clock();

    // movie_ratings: for each movie, the users and ratings that it has
    std::vector<std::unordered_map<int, int>> movie_ratings;
    std::vector<std::unordered_map<int, int>> user_ratings;
    LoadRatings(rating_name, n_movies, n_users, movie_ratings, user_ratings, is100k);

    t = clock() - t;
    printf("Loading dataset took %f seconds.\n",((float)t)/CLOCKS_PER_SEC);
    
    // Singular run of KNN
    if (gpu != 2)
    {
        // Compute the distance metric
        float * movie_distances;
        if (gpu == 1) {
            movie_distances = new float [n_movies * n_movies];
            cuda_call_pearson_kernel(65535, 1024, movie_ratings, user_ratings,
                                     n_movies, n_users, movie_distances);
        }
        else {
            pearson_distances(movie_ratings, user_ratings, n_movies, &movie_distances, is100k);
        }

        if (verbose)
            print_array("MovieLens Movie Distances", &movie_distances, 1, 100);
        t = clock() - t;
        printf("Computing Distances for size %s took %f seconds.\n",
            dataset_size.c_str(), ((float)t)/CLOCKS_PER_SEC);

        // Find the k-nearest neighbors
        float * knn_predictions;
        if (gpu == 1) {
            int pred_users = 900;
            int pred_movies = 1500;
            knn_predictions = new float[pred_users * pred_movies];
            // reduced blocks, threads_per_block due to gpu memory limitations
            cuda_call_prediction_kernel(32, 32, user_ratings, movie_distances, 
                                        k_val, n_movies, n_users, knn_predictions, pred_movies, pred_users, true);
        }
        else {
            knn(user_ratings, &movie_distances, k_val, n_movies, n_users, &knn_predictions, true);
        }
        if (verbose)
            print_array("MovieLens Pearson KNN Predictions", &knn_predictions, 1, 100);

        t = clock() - t;
        printf("KNN Predictions with Pearson Correlation metric for size %s took %f seconds.\n",
            dataset_size.c_str(), ((float)t)/CLOCKS_PER_SEC);


        delete[] movie_distances;
        delete[] knn_predictions;
    }
    // Test to compare the GPU results against the CPU results
    else {
        float * cpu_distances;
        float * gpu_distances;

        // GPU metric computation
        gpu_distances = new float [n_movies * n_movies];
        cuda_call_pearson_kernel(65535, 1024, movie_ratings, user_ratings,
                                 n_movies, n_users, gpu_distances);
        t = clock() - t;
        printf("Computing GPU Distances for size %s took %f seconds.\n",
         dataset_size.c_str(), ((float)t)/CLOCKS_PER_SEC);

        // CPU metric computation
        pearson_distances(movie_ratings, user_ratings, n_movies, &cpu_distances, is100k);
        t = clock() - t;
        printf("Computing CPU Distances for size %s took %f seconds.\n",
            dataset_size.c_str(), ((float)t)/CLOCKS_PER_SEC);

        // Make sure the distances are the same
        compare_arrays(cpu_distances, gpu_distances, n_movies * n_movies);
        t = clock() - t;
        printf("Checking CPU and GPU distances are the same for size %s took %f seconds.\n",
            dataset_size.c_str(), ((float)t)/CLOCKS_PER_SEC);
        
        
        // Find the k-nearest neighbors
        float * cpu_knn_predictions;
        float * gpu_knn_predictions;
        
        // GPU knn prediction
        int pred_users = 900;
        int pred_movies = 1500;
        gpu_knn_predictions = new float[pred_users * pred_movies];
        cuda_call_prediction_kernel(32, 32, user_ratings, gpu_distances, 
                                    k_val, n_movies, n_users, gpu_knn_predictions, pred_movies, pred_users, true);
        t = clock() - t;
        printf("Computing GPU predictions for size %s took %f seconds.\n",
            dataset_size.c_str(), ((float)t)/CLOCKS_PER_SEC);
        
        // CPU knn prediction
        knn(user_ratings, &cpu_distances, k_val, n_movies, n_users, &cpu_knn_predictions, true);
        t = clock() - t;
        printf("Computing CPU predictions for size %s took %f seconds.\n",
            dataset_size.c_str(), ((float)t)/CLOCKS_PER_SEC);
        
        // Print predictions side by side at random indices for comparison
        printf("Comparing CPU and GPU predictions\n");
        print_random_entries(cpu_knn_predictions, gpu_knn_predictions, 100, pred_users * pred_movies);

        delete[] cpu_distances;
        delete[] gpu_distances;
        delete[] cpu_knn_predictions;
        delete[] gpu_knn_predictions;
    }
}

/*
KNN Calculations with Euclidean Distance Metric
*/
void euclidean_movie_lens(bool is100k, int gpu, int k_val, bool verbose)
{
    // Clock the performance on the euclidean metric
    clock_t t;
    t = clock();

    // MovieLens Dataset defaults
    int n_movies = 27278;
    int n_users = 138493;
    std::string dir_name = "ml-20m/";
    std::string dataset_size = "20 million";
    if (is100k) {
        dir_name = "ml-100k/";
        dataset_size = "100 thousand";
        n_movies = 1682;
        n_users = 943;
    }

    std::vector<std::string> genres = {
        "Action", "Adventure", "Animation", "Children's",
        "Comedy", "Crime", "Documentary", "Drama",
        "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller",
        "War", "Western", "(no genres listed)"
    };
    int movie_size = genres.size(); //+ 1;
    if (is100k) {
        movie_size = 19; // 20
    }

    // Load the dataset
    unsigned int * movie_data;
    LoadGenres(is100k, dir_name, n_movies, movie_size, &movie_data, genres);

    std::string rating_name = dir_name + "ratings.csv";
    if (is100k)
        rating_name = dir_name + "u.data";
    std::vector<std::unordered_map<int, int>> movie_ratings;
    std::vector<std::unordered_map<int, int>> user_ratings;
    LoadRatings(rating_name, n_movies, n_users, movie_ratings, user_ratings, is100k);
    //if (verbose)
    //print_ratings(n_movies, movie_ratings);
    //print_ratings(n_users, user_ratings);

    if (verbose)
        print_array("MovieLens Euclidean Movies", &movie_data, 1, 100);
    t = clock() - t;
    printf("Loading Movie Data for size %s took %f seconds.\n",
        dataset_size.c_str(), ((float)t)/CLOCKS_PER_SEC);


    // Singular run of KNN
    if (gpu != 2)
    {
        // Compute the distance metric
        float * movie_distances;
        if (gpu == 1) {
            movie_distances = new float [n_movies * n_movies];
            cuda_call_euclidean_kernel(65535, 1024, movie_data, n_movies,
                                       movie_size, movie_distances);
        }
        else {
            euclidean_distances(&movie_data, n_movies, movie_size, &movie_distances);
        }

        if (verbose)
            print_array("MovieLens Movie Distances", &movie_distances, n_movies, n_movies);
        t = clock() - t;
        printf("Computing Distances for size %s took %f seconds.\n",
            dataset_size.c_str(), ((float)t)/CLOCKS_PER_SEC);
            
        // Find the k-nearest neighbors
        float * knn_predictions;
        if (gpu == 1) {
            int pred_users = 900;
            int pred_movies = 1500;
            knn_predictions = new float[pred_users * pred_movies];
            // reduced blocks, threads_per_block due to gpu memory limitations
            cuda_call_prediction_kernel(32, 32, user_ratings, movie_distances, 
                                        k_val, n_movies, n_users, knn_predictions, pred_movies, pred_users, true);
        }
        else {
            knn(user_ratings, &movie_distances, k_val, n_movies, n_users, &knn_predictions, true);
        }
        
        if (verbose)
            print_array("MovieLens Euclidean KNN Predictions", &knn_predictions, 1, 100);

        t = clock() - t;
        printf("KNN Predictions with Euclidean distance metric for size %s took %f seconds.\n",
            dataset_size.c_str(), ((float)t)/CLOCKS_PER_SEC);


        delete[] movie_distances;
        delete[] knn_predictions;
    }
    // Test to compare the GPU results against the CPU results
    else {
        float * cpu_distances;
        float * gpu_distances;

        // GPU metric computation
        gpu_distances = new float [n_movies * n_movies];
        cuda_call_euclidean_kernel(65535, 1024, movie_data, n_movies,
                                   movie_size, gpu_distances);
        t = clock() - t;
        printf("Computing GPU Distances for size %s took %f seconds.\n",
         dataset_size.c_str(), ((float)t)/CLOCKS_PER_SEC);

        // CPU metric computation
        euclidean_distances(&movie_data, n_movies, movie_size, &cpu_distances);
        t = clock() - t;
        printf("Computing CPU Distances for size %s took %f seconds.\n",
            dataset_size.c_str(), ((float)t)/CLOCKS_PER_SEC);

        // Make sure the distances are the same
        compare_arrays(cpu_distances, gpu_distances, n_movies * n_movies);
        t = clock() - t;
        printf("Checking CPU and GPU distances are the same for size %s took %f seconds.\n",
            dataset_size.c_str(), ((float)t)/CLOCKS_PER_SEC);

        // Find the k-nearest neighbors
        float * cpu_knn_predictions;
        float * gpu_knn_predictions;
        
        // GPU knn prediction
        int pred_users = 900;
        int pred_movies = 1500;
        gpu_knn_predictions = new float[pred_users * pred_movies];
        cuda_call_prediction_kernel(32, 32, user_ratings, gpu_distances, 
                                    k_val, n_movies, n_users, gpu_knn_predictions, pred_movies, pred_users, true);
        t = clock() - t;
        printf("Computing GPU predictions for size %s took %f seconds.\n",
            dataset_size.c_str(), ((float)t)/CLOCKS_PER_SEC);
        
        // CPU knn prediction
        knn(user_ratings, &cpu_distances, k_val, n_movies, n_users, &cpu_knn_predictions, true);
        t = clock() - t;
        printf("Computing CPU predictions for size %s took %f seconds.\n",
            dataset_size.c_str(), ((float)t)/CLOCKS_PER_SEC);
        
        // Print randomly selected predicitons to compare cpu and gpu output
        printf("Comparing CPU and GPU predictions\n");
        print_random_entries(cpu_knn_predictions, gpu_knn_predictions, 100, pred_users * pred_movies);

        delete[] cpu_distances;
        delete[] gpu_distances;
        delete[] cpu_knn_predictions;
        delete[] gpu_knn_predictions;
    }

    delete[] movie_data;
}

// Prints the full argument list and filters to respective metric function
void movie_lens(int *args)
{
    std::string dataset_size = "20 million";
    if (args[0])
        dataset_size = "100 thousand";

    std::string gpu_cpu = "just the CPU";
    if (args[1])
        gpu_cpu = "GPU acceleration";

    std::string metric = "euclidean distance";
    if (args[2])
        metric = "pearson correlation";

    std::cout << "KNN for Movie-Lens of size " << dataset_size <<
        " with " << gpu_cpu << " and metric of " << metric <<
        " and K = " << args[3] << std::endl;

    if (args[2])
        pearson_movie_lens(args[0], args[1], args[3], args[4]);
    else
        euclidean_movie_lens(args[0], args[1], args[3], args[4]);
}


int main(int argc, char **argv)
{
    // Usage: ./knn is100k gpu pearson
    // For example: ./knn 0 1 0 , ./knn 1 , ./knn 0
    // Or just ./knn for defaults

    // is100k = 0,1;    : Use the 100k or the 20mil datasets
    // gpu = 0, 1, 2;   : GPU-acceleration flag, 2 for correction test
    // pearson = 0,1;   : Use Pearson Correlation or Euclidean distance metric
    // k_val = 3        : Desired value of k to search for
    // verbose = 0, 1   : Whether to log debugging output or not
    int *args = new int[5];
    args[0] = 0;    // is100k
    args[1] = 1;    // gpu
    args[2] = 0;    // pearson
    args[3] = 3;    // k_val
    args[4] = 0;    // verbose

    if (argc > 6)
    {
        std::cout << "Too many arguments. Usage: ./knn is100k gpu pearson k_val verbose" << std::endl;
    }
    else
    {
        for (int i = 1; i < argc; i++)
        {
            args[i-1] = std::atoi(argv[i]);
        }
    }

    movie_lens(args);

    return 0;
}
