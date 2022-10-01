#pragma once

std::vector<int> runCudaBasic(const matrix &data, matrix &centroids, std::vector<int> &flags, int maxIterations, double threshold);

std::vector<int> runCudaShmem(const matrix &data, matrix &centroids, std::vector<int> &flags, int maxIterations, double threshold);

std::vector<int> runThrust(const matrix &data, matrix &centroids, std::vector<int> &flags, int maxIterations, double threshold);
