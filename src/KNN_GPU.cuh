#ifndef KNN_GPU_DEVICE_CUH
#define KNN_GPU_DEVICE_CUH

#include <vector>
#include <unordered_map>
#include "cuda_header.cuh"

int cuda_call_euclidean_kernel(const unsigned int blocks,
                               const unsigned int threads_per_block,
                               const unsigned int *movie_data,
                               const int n_movies,
                               const int n_features,
                               float *movie_distances);

int cuda_call_pearson_kernel(const unsigned int blocks,
                             const unsigned int threads_per_block,
                             std::vector<std::unordered_map<int, int>> &movie_ratings,
                             std::vector<std::unordered_map<int, int>> &user_ratings,
                             const int n_movies,
                             const int n_users,
                             float *movie_distances);
                             
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
                                const bool pearson=false);

#endif