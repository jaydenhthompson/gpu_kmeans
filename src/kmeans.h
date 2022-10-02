#pragma once

#include "helpers.h"

#include <vector>

std::vector<float> runCudaBasic(const matrix &data, matrix &centroids, std::vector<int> &flags, int maxIterations, double threshold);

std::vector<float> runCudaShmem(const matrix &data, matrix &centroids, std::vector<int> &flags, int maxIterations, double threshold);

std::vector<float> runThrust(const matrix &data, matrix &centroids, std::vector<int> &flags, int maxIterations, double threshold);
