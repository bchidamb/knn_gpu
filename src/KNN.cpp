#include <math.h>
#include <time.h>
#include <bits/stdc++.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>

// According to KNN blog post
// http://dmnewbie.blogspot.com/2009/06/calculating-316-million-movie.html
struct PearsonIntermediate {
  //from all viewers who rated both movie X and Y.
    float x; //sum of ratings for movie X
    float y; //sum of ratings for movie Y
    float xy; //sum of product of ratings for movies X and Y
    float xx; //sum of square of ratings for movie X
    float yy; //sum of square of ratings for movie Y
    unsigned int cnt; //number of viewers who rated both movies
  };

// Structure to handle sorting in place
struct sort_struct
{
    int movie;
    float weight;
    float rating;
};
bool euclidean_compare(sort_struct lhs, sort_struct rhs) {return lhs.weight < rhs.weight; }

bool pearson_compare(sort_struct lhs, sort_struct rhs) {return lhs.weight > rhs.weight; }


// Computes the average rating for a given movie movie
float compute_mean(std::unordered_map<int, int> &single_movie_ratings)
{
    float m = 0.0;
    int num = 0;
    for (auto it : single_movie_ratings)
    {
        m += it.second;
        num += 1;
    }
    m = m / (float) num;
    return m;
}

// Computes the standard deviation of a given movie
// std(X) = sqrt(E[(X - ux)^2])
float compute_std_dev(std::unordered_map<int, int> &single_movie_ratings, float &m)
{
    float std_dev = 0.0;
    int num = 0;
    for (auto it : single_movie_ratings)
    {
        std_dev += pow(it.second - m, 2);
        num += 1;
    }
    std_dev = sqrt(std_dev / (float) num);
    return std_dev;
}

// Computes the correlation between two movies i and j
// Deprecated now that we are doing accumulation method
// to calculate the Pearson correlations
// cov(X,Y) = E[(X - ux)(Y - uy)]
float correlation(std::unordered_map<int, int> &i_ratings,
    std::unordered_map<int, int> &j_ratings,
    float x_mean, float x_std_dev,
    float y_mean, float y_std_dev)
{
    float correlation = 0.0;
    // Correlation is 0 if the standard deviation is 0
    if (x_std_dev == 0 || y_std_dev == 0)
    {
        return correlation;
    }

    float delX = 0.0;
    float delY = 0.0;

    int num = 0;
    for (auto it: i_ratings)
    {
        // Only compute correlation among shared entries
        if (j_ratings.find(it.first) != j_ratings.end())
        {
            delX = it.second - x_mean;
            for (auto jt : j_ratings)
            {
                delY = jt.second - y_mean;
                correlation += (delX * delY);
                num += 1;
            }
        }
    }
    correlation = correlation / (num * x_std_dev * y_std_dev);

    return correlation;
}

// Using Pearson correlation coefficient as opposed to euclidean distance
// Pearson correlation is defined as cov(X, Y) / std(X)*std(Y)
// std(X) = sqrt(E[(X - ux)^2])
// cov(X,Y) = E[(X - ux)(Y - uy)]
void pearson_distances(std::vector<std::unordered_map<int, int>> &movie_ratings,
    std::vector<std::unordered_map<int, int>> &user_ratings,
    const int &n_movies, float ** movie_distances, bool is100k)
{
    // For each movie, calculate the expected rating baseline mean
    // And standard deviation
    float * means = new float[n_movies];
    float * std_devs = new float[n_movies];
    for (int i = 0; i < n_movies; i++)
    {
        means[i] = compute_mean(movie_ratings[i]);
        std_devs[i] = compute_std_dev(movie_ratings[i], means[i]);
    }

    int numZero = 0;
    for (int i = 0; i < n_movies; i++)
    {
        //std::cout << "Movie " << i+1 << " has mean " << means[i] << " and stdev " << std_devs[i] << std::endl;
        if (std_devs[i] == 0) {
            numZero += 1;
        }
    }
    std::cout << "Number of movies with standard deviation of 0: " << numZero << std::endl;

    // Psuedo-code from:
    // http://dmnewbie.blogspot.com/2007/09/greater-collaborative-filtering.html
    (*movie_distances) = new float[n_movies * n_movies];
    PearsonIntermediate one_result[n_movies];

    int percent = 0;
    for (int i = 0; i < n_movies; i++)
    {
        // Initialize the single result intermediate to 0
        for (int j = 0; j < n_movies; j++)
        {
            one_result[j].x = 0;
            one_result[j].y = 0;
            one_result[j].xy = 0;
            one_result[j].xx = 0;
            one_result[j].yy = 0;
            one_result[j].cnt = 0;
        }

        // Iterate through all users that rated movie i
        for (auto u_rating : movie_ratings[i])
        {
            int user = u_rating.first;
            float rating_i = u_rating.second;

            // Iterate through all movies that user rated and update intermediate
            for (auto m_rating : user_ratings[user])
            {
                int m = m_rating.first;
                float rating_j = m_rating.second;
                
                // Update the intermediaries
                one_result[m].x += rating_i;
                one_result[m].y += rating_j;
                one_result[m].xy += rating_i * rating_j;
                one_result[m].xx += rating_i * rating_i;
                one_result[m].yy += rating_j * rating_j;
                one_result[m].cnt += 1;
            }
            
            
        }
        // Go through the array to calculate correlations
        for (int j = 0; j < n_movies; j++)
        {
            float x = one_result[j].x;
            float y = one_result[j].y;
            float xy = one_result[j].xy;
            float xx = one_result[j].xx;
            float yy = one_result[j].yy;
            float cnt = one_result[j].cnt;

            //printf("Movie %i to Movie %i: %f, %f, %f, %f, %f, %f\n", i, j, x, y, xy, xx, yy, cnt);

            if (i == j) {
                (*movie_distances)[i * n_movies + j] = 1.0;

            } else  {
                if (cnt == 0)
                {
                    (*movie_distances)[i * n_movies + j] = 0;
                }
                else
                {
                    float result = (cnt * xy - x * y) /
                        (sqrt(cnt * xx - x*x) * sqrt(cnt * yy - y*y));

                    // Check that result is not NaN
                    // https://stackoverflow.com/questions/570669/checking-if-a-double-or-float-is-nan-in-c
                    if (result != result)
                        result = 0.0;
                    //std::cout << "Movie " << i << "to Movie " << j << "has Pearson of " << result << std::endl;
                    (*movie_distances)[i * n_movies + j] = result;
                }
            }
        }
        if (i % int(n_movies/100) == 0)
        {
            percent += 1;
            std::cout << "Pearson: " << percent << "% complete." << std::endl;
        }

    }
}

// Distance metric that takes the euclidean distance between
// each pair of movies based on the genres that each movie falls under
void euclidean_distances(unsigned int **movie_data, const int &n_movies,
    const int &movie_size, float **movie_distances)
{
    *movie_distances = new float[n_movies * n_movies];
    float dist;
    for (int i = 0; i < n_movies; i++)
    {
       for (int j = i; j < n_movies; j++)
       {
           if (i == j) {
               (*movie_distances)[i * n_movies + j] = -1.0;
           } else {
               // Get the squared difference
               dist = 0;
               for (int k = 0; k < movie_size; k++)
               {
                   dist += pow(float((*movie_data)[i * movie_size + k])
                       - float((*movie_data)[j * movie_size + k]), 2);
               }
               (*movie_distances)[i * n_movies + j] = dist;
               (*movie_distances)[j * n_movies + i] = dist;
           }
       }
    }
}

// For a given movie, user pair, returns the KNN prediction
float predict(int &user, int &movie, int &n_movies, int &k_val,
    float ** movie_distances,
    std::unordered_map<int, int> &u_ratings, bool pearson)
{

    // Find the k nearest neighboring movie ratings
    // that the same user has rated
    // and take an average to return the prediction
    sort_struct *mu_row = new sort_struct[u_ratings.size()];
    sort_struct aStruct;

    int i = 0;
    // Look at all other movies that current user has rated
    // And create a sortable structure of movie, weight (metric), and rating
    for (auto movie_rating : u_ratings)
    {
        aStruct.movie = movie_rating.first;
        aStruct.weight = (*movie_distances)[movie * n_movies + aStruct.movie];
        if (!pearson)
            aStruct.weight = 1 / (aStruct.weight + 1);
        aStruct.rating = movie_rating.second;
        mu_row[i] = aStruct;
        i++;
    }

    // Sort the structure of weights/ratings based on the weight metric
    if (pearson)
    {
        std::sort(mu_row, mu_row + u_ratings.size(), pearson_compare);
    } else {
        std::sort(mu_row, mu_row + u_ratings.size(), euclidean_compare);
    }

    // Return a weighted average of the k-nearest neighbors
    float weight_sum = 0.0;
    float result = 0.0;
    for (int j = 0; j < k_val; j++)
    {
        result += mu_row[j].weight * mu_row[j].rating;
        weight_sum += mu_row[j].weight;
    }
    if (weight_sum == 0.0)
        return 0.0;
    result = result / weight_sum;
    //std::cout << "User " << user << " Movie " << movie << ": " << result << std::endl;
    return result;
}


// KNN with an input distance metric
void knn(std::vector<std::unordered_map<int, int>> &user_ratings,
    float **movie_distances, int &k_val, int &n_movies, int &n_users,
    float **knn_predictions, bool pearson = false)
{
    // Take a sample of about 100k, movie-user pairs as a test set
    int pred_movies = 1500;
    int pred_users = 900;

    //*knn_predictions = new float[n_movies * k_val];
    *knn_predictions = new float[pred_users * pred_movies];
    int percent = 0;
    for (int i = 0; i < pred_users; i++)
    {
        std::unordered_map<int, int> u_ratings = user_ratings[i];

        // Make sure user has enough ratings to be at least k
        if (u_ratings.size() >= k_val) {
            for (int j = 0; j < pred_movies; j++)
            {
                (*knn_predictions)[i * pred_movies + j] = predict(i, j,
                    n_movies, k_val, movie_distances, u_ratings, pearson);
            }
        }
        else {
            for (int j = 0; j < pred_movies; j++)
            {
                (*knn_predictions)[i * pred_movies + j] = 0;
            }
        }

        if (i % int(pred_users/100) == 0)
        {
            percent += 1;
            std::cout << "KNN Prediction is " << percent << "% complete." << std::endl;
        }
    }
}
