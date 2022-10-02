#pragma once

#include "helpers.h"

#include <cmath>
#include <chrono>
#include <vector>

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

    auto movement = calculateMovement(centroids, newCentroids);

    centroids = newCentroids;
    return movement;
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

std::vector<float> runSequentialKMeans(const matrix &data, matrix &centroids, std::vector<int> &flags, int maxIterations, double threshold)
{
    double convergence;
    std::vector<float> iterTimes;
    for (int i = 0; i < maxIterations; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        calculateFlags(data, centroids, flags);
        convergence = recalculateCentroids(data, flags, centroids);
        auto end = std::chrono::high_resolution_clock::now();

        int microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        iterTimes.push_back(static_cast<float>(microseconds)/1000.0);
        if (convergence <= threshold)
            break;
    }
    return iterTimes;
}