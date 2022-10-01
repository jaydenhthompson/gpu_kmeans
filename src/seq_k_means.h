#pragma once

#include "helpers.h"

#include <vector>

double recalculateCentroids(const matrix &data, const std::vector<int> &flags, matrix &centroids);

void calculateFlags(const matrix &data, const matrix &centroids, std::vector<int> &flags);

double calculateEuclidean(const std::vector<double> &a, const std::vector<double> &b);

void runSequentialKMeans(const matrix &data, matrix &centroids, std::vector<int> &flags, int maxIterations, double threshold);