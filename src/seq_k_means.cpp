#include "seq_k_means.h"

#include <cmath>

double recalculateCentroids(const matrix &data, const std::vector<int> &flags, matrix &centroids)
{
    matrix newCentroids(centroids.size(), std::vector<double>(centroids[0].size(), 0.0));
    std::vector<int> numAssigned(centroids.size(), 0);
    for (size_t i = 0; i < data.size(); i++)
    {
        int targetCentroid = flags[i];
        numAssigned[targetCentroid]++;
        for(size_t j = 0; j < data[i].size(); j++)
        {
            newCentroids[targetCentroid][j] += data[i][j];
        }
    }

    for(size_t i = 0; i < newCentroids.size(); i++)
    {
        for(auto & e : newCentroids[i])
        {
            e /= numAssigned[i]; 
        }
    }

    double maxMovement = 0.0;
    for(size_t i = 0; i < centroids.size(); i++)
    {
        maxMovement = std::max(maxMovement, calculateEuclidean(centroids[i], newCentroids[i]));
    }

    centroids = newCentroids;
    return maxMovement;
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

void calculateFlags(const matrix &data, const matrix &centroids, std::vector<int> &flags)
{
    for(size_t i = 0; i < data.size(); i++)
    {
        int assigned = 0;
        double minDist = std::numeric_limits<double>::max();
        for (size_t j = 0; j < centroids.size(); j++)
        {
            auto dist = calculateEuclidean(data[i], centroids[j]);
            if (dist < minDist)
            {
                minDist = dist;
                assigned = j;
            }
        }
        flags[i] = assigned;
    }
}

void runSequentialKMeans(const matrix &data, matrix &centroids, std::vector<int> &flags, int maxIterations, double threshold)
{
    double convergence;
    for(int i = 0; i < maxIterations; i++)
    {
        calculateFlags(data, centroids, flags);
        convergence = recalculateCentroids(data, flags, centroids);
        if (convergence <= threshold)
            return;
    }
}