#include "helpers.h"

#include <cstdlib>

matrix getRandomCentroids(int num_dimensions, int num_centroids, uint seed)
{
    std::srand(seed);
    matrix centroids(num_centroids, std::vector<double>(num_dimensions));


    return centroids;
}