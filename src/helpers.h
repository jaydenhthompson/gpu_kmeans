#pragma once

#include <cstdlib>
#include <vector>

typedef std::vector<std::vector<double>> matrix;

matrix getRandomCentroids(const matrix& points, int num_dimensions, int num_centroids, uint seed);