# K-Nearest Neighbors algorithm implemented in python
import os
import time
import csv

def compute_distances(movie_data):
    """
    Takes a movie matrix and creates a n x n distance matrix
    """
    n = len(movie_data)
    movie_size = len(movie_data[0])
    movie_distances = [[0 for i in range(n)] for j in range(n)]

    for i in range(n):
        for j in range(i, n):
            if i == j:
                movie_distances[i][j] = -1.0
            else:
                dist = 0
                for k in range(1, movie_size):
                    dist += float(movie_data[i][k] - movie_data[j][k]) ** 2
                movie_distances[i][j] = dist
                movie_distances[j][i] = dist

    return movie_distances

def knn(movie_distances, k = 3):
    """
    For each movie, finds the k nearest neighbors given the movie distances
    """
    n = len(movie_distances)
    k_nearest_neighbors = [[0 for i in range(k)] for j in range(n)]
    movie_row = [(0, 0) for i in range(n)]

    for i in range(n):
        for j in range(n):
            movie_row[j] = (j, movie_distances[i][j])

        movie_row.sort(key=lambda tup: tup[1])

        for j in range(1, k + 1):
            k_nearest_neighbors[i][j-1] = movie_row[j][0]


    # Print out to verify results
    # print(k_nearest_neighbors[n-1])
    # for i in range(n):
    #     print(movie_row[i])
    return k_nearest_neighbors

if __name__ == "__main__":

    print("KNN CPU Implementation in Python")
    t0 = time.time()

    # Number of categories
    genres = [ \
        "Action", "Adventure", "Animation", "Children's", \
        "Comedy", "Crime", "Documentary", "Drama", \
        "Fantasy", "Film-Noir", "Horror", "Musical", \
        "Mystery", "Romance", "Sci-Fi", "Thriller", \
        "War", "Western", "(no genres listed)"]
    movie_size = len(genres) + 1

    # movies = pd.read_csv("../ml-20m/movies.csv")
    # n = len(movies.index)
    n = 27278
    movies = ""
    movie_data = [[0 for i in range(movie_size)] for j in range(n)]
    index = -1
    with open("../ml-20m/movies.csv", "rb") as movieFile:
        movies = csv.reader(movieFile)

        for row in movies:
            if index != -1:
                movie_data[index][0] = int(row[0])

                row_genres = row[2].split("|")
                for i in range(len(genres)):
                    if genres[i] in row_genres:
                        movie_data[index][i+1] = 1
            index += 1

    # Build the matrix of movies
    # movie_data = [[0 for i in range(movie_size)] for j in range(n)]
    # for index, row in movies.iterrows():
    #     movie_data[index][0] = int(row["movieId"])
    #
    #     row_genres = row["genres"].split("|")
    #     for i in range(len(genres)):
    #         if genres[i] in row_genres:
    #             movie_data[index][i+1] = 1

    #print(movie_data[n-1])

    print("Loading movies took " + str(time.time() - t0) + " seconds.")
    t0 = time.time()

    # Compute distances
    movie_distances = compute_distances(movie_data)

    print("Computing distances took " + str(time.time() - t0) + " seconds.")
    t0 = time.time()

    # Get the k-nearest neighbors
    k_nearest_neighbors = knn(movie_distances)

    print("Computing KNN took " + str(time.time() - t0) + " seconds.")
    t0 = time.time()
