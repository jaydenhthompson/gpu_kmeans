#include "helpers.h"

#include <cstdlib>

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
    next = seed;
}

matrix getRandomCentroids(const matrix& points, int num_dimensions, int num_centroids, uint seed)
{
    kmeans_srand(seed); 
    matrix centers(num_centroids, std::vector<double>(num_dimensions));
    for (int i = 0; i < num_centroids; i++)
    {
        int index = kmeans_rand() % points.size();

        centers[i] = points[index];
    }

    return centers;
}

double calculateEuclidean(const std::vector<double> &a, const std::vector<double> &b)
{
    double dist = 0.0;
    for(size_t i = 0; i < a.size(); i++)
    {
        dist += std::pow(a[i] - b[i], 2);
    }
    return std::sqrt(dist);
}

double calculateMovement(const matrix &a, const matrix &b)
{
    double maxMovement = 0.0;
    for(size_t i = 0; i < a.size(); i++)
    {
        maxMovement = std::max(maxMovement, calculateEuclidean(a[i], b[i]));
    }
    return maxMovement
}