#pragma once

#include "helpers.h"

#include <vector>

std::vector<float> runCudaBasic(const std::vector<double> &data, std::vector<double> &centroids, std::vector<int> &flags, int dimensions, int numData, int numClusters, int maxIterations, double threshold);

std::vector<float> runCudaShmem(const std::vector<double> &data, std::vector<double> &centroids, std::vector<int> &flags, int maxIterations, double threshold);

std::vector<float> runThrust(const std::vector<double> &data, std::vector<double> &centroids, std::vector<int> &flags, int maxIterations, double threshold);
