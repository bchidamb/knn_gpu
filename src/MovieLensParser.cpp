#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>

bool is_number(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

std::vector<std::string> split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

void LoadRatings(std::string ratings_fname, const int &n_movies, const int &n_users,
    std::vector<std::unordered_map<int, int>> &movie_ratings,
    std::vector<std::unordered_map<int, int>> &user_ratings, bool is100k = true)
{
    // movie_ratings: for each movie, the users and ratings that it has

    movie_ratings.reserve(n_movies);
    std::unordered_map<int,int> blank;
    for (int i = 0; i < n_movies; i++)
    {
        movie_ratings[i] = blank;
    }

    // user_ratings: for each user, the movie ids and ratings

    user_ratings.reserve(n_users);
    for (int i = 0; i < n_users; i++)
    {
        user_ratings[i] = blank;
    }

    std::ifstream inputFile(ratings_fname);
    if (inputFile.is_open())
    {
        std::vector<std::string> tokens;
        std::string line;
        if (!is100k)
            std::getline(inputFile, line);

        int movie_index;
        int user_no;
        int rating;
        while (std::getline(inputFile, line))
        {
            //std::cout << line << std::endl;
            if (is100k)
            {
                tokens = split(line, "\t");
            } else {
                tokens = split(line, ",");
            }
            if (tokens.size() == 4)
            {
                
                user_no = std::stoi(tokens[0]) - 1;
                movie_index = std::stoi(tokens[1]) - 1;
                rating = std::stoi(tokens[2]);

                //std::cout << movie_index << "," << user_no << "," << rating << std::endl;
                if (movie_index < n_movies) {
                    movie_ratings[movie_index][user_no] = rating;
                    user_ratings[user_no][movie_index] = rating;
                }
            }
        }
    }

}


void LoadGenres100k(std::string movie_fname, int &n_movies,
    int &movie_size, unsigned int **movie_data,
    bool movie = true, std::unordered_map<std::string, int> occupations = {})
{
    // movie_fname      :
    // n_movies (1682)  :
    // movie_size (20)  :
    // movie_data       :
    // movie (optional) :
    // occuptions (opt) :

    int n_occupations = occupations.size();

    *movie_data = new unsigned int[n_movies * movie_size];
    std::ifstream inputFile(movie_fname);

    if (inputFile.is_open())
    {
        std::string line;
        int genre_count;
        int movie_count = 0;
        std::string i;

        // Read the movie entry into memory
        while (movie_count < n_movies && std::getline(inputFile, line))
        {
            std::istringstream iss(line);
            genre_count = 0;
            while (std::getline(iss, i, '|'))
            {
                // If loading user data, then check if string is an occupation
                if (!movie)
                {
                    // Convert male female to two different attributes
                    if (i == "M" || i == "F")
                    {
                        if (i == "M") {
                            (*movie_data)[(movie_count * movie_size) + genre_count - 1] = 1;
                            genre_count++;
                            (*movie_data)[(movie_count * movie_size) + genre_count - 1] = 0;
                            genre_count++;
                        } else {
                            (*movie_data)[(movie_count * movie_size) + genre_count - 1] = 0;
                            genre_count++;
                            (*movie_data)[(movie_count * movie_size) + genre_count - 1] = 1;
                            genre_count++;
                        }
                    }

                    // One-hot encoding the occupations like any other movie category
                    if (occupations.find(i) != occupations.end())
                    {
                        // Set this occupation category to
                        int occ_ind = occupations[i];

                        // Turn all other occupations to 0 except for this
                        for (int j = 0; j < n_occupations; j++)
                        {
                            if (j != occ_ind)
                            {
                                (*movie_data)[(movie_count * movie_size) + genre_count - 1] = 0;
                            } else {
                                (*movie_data)[(movie_count * movie_size) + genre_count - 1] = 1;
                            }
                            genre_count++;
                        }
                    }
                }

                if (is_number(i))
                {
                    // Update the entry for the given category
                    if (genre_count > 0)
                        (*movie_data)[(movie_count * movie_size) + genre_count - 1] = abs(std::atoi(i.c_str()));
                    genre_count++;
                }
            }
            movie_count++;
        }
        inputFile.close();
    }
}


void LoadGenres20mil(std::string movie_fname, int &n_movies,
    unsigned int **movie_data, std::vector<std::string> genres)
{

    // n_movies: 27278
    int n_genres = genres.size();
    int movie_size = n_genres; //+ 1; // number of genres + movie id

    *movie_data = new unsigned int[n_movies * movie_size];
    std::ifstream inputFile(movie_fname);
    if (inputFile.is_open())
    {
        std::string line;
        std::vector<std::string> tokens;
        std::vector<std::string> glist;
        int movie_count = 0;
        while (std::getline(inputFile, line))
        {
            tokens = split(line, ",");

            // Movie ID
            if (is_number(tokens[0]))
            {
                // Update the movie id
                //(*movie_data)[(movie_count * movie_size) + 0] = abs(std::atoi(tokens[0].c_str()));


                // Genre unpacking
                glist = split(tokens[tokens.size() - 1], "|");
                std::set<std::string> genre_set (glist.begin(), glist.end());
                for (int i = 0; i < n_genres; i++)
                {
                    if (genre_set.find(genres.at(i)) != genre_set.end())
                    {
                        (*movie_data)[(movie_count * movie_size) + i] = 1; // + 1
                    } else {
                        (*movie_data)[(movie_count * movie_size) + i] = 0; // + 1
                    }
                }
            }

            movie_count++;
        }
    }
    inputFile.close();
}

void LoadGenres(bool is100k, std::string dir_name, int &n_movies,
    int &movie_size, unsigned int **movie_data, std::vector<std::string> genres)
{
    if (is100k)
    {
        LoadGenres100k(dir_name + "u.item", n_movies, movie_size, movie_data);
    }
    else {
        LoadGenres20mil(dir_name + "movies.csv", n_movies, movie_data, genres);
    }
}

// Function to load data into movie and user dataset pointers
// alongside number of categories, e.t.c
// Make sure these things are int pointers

// Movie: movie_id, and then 19 ints of either 0 or 1
// corresponding to the genres that movies are a part of
// pointer array of 20 ints

// User: user_id, age, gender, occupation (turn into int), zip code (roughly geographical)
// another linearized pointer array of ints??

// Store 2d matrix of distances between movies and users in shared memory and update
// but won't we get bank conflicts then???
// maybe not if we don't do a dynamic programming approach
// but then we still have a lot of repeated computation
