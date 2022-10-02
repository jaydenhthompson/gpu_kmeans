#pragma once

#include <cstdlib>
#include <vector>

typedef std::vector<std::vector<double>> matrix;

matrix getRandomCentroids(const matrix& points, int num_dimensions, int num_centroids, uint seed);

double calculateEuclidean(const std::vector<double> &a, const std::vector<double> &b);

double calculateMovement(const matrix &a, const matrix &b);

