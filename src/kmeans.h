#pragma once

#include "helpers.h"

#include <vector>

std::vector<float> runCuda(const std::vector<double> &data, std::vector<double> &centroids, std::vector<int> &flags, int option, int dimensions, int numData, int numClusters, int maxIterations, double threshold);
