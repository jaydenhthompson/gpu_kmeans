#include "helpers.h"
#include "kmeans.h"

// CUDA Includes
#include <cuda_runtime.h>

#include <vector>

__device__ double euclideanDistance(double *a, double *b, int dim)
{

}

std::vector<int> runCudaBasic(const matrix &data, matrix &centroids, std::vector<int> &flags, int maxIterations, double threshold)
{
    double *d_data = nullptr;
    int dataSize = data.size() * data[0].size() * sizeof(double);
    cudaMalloc((void**)&d_data, dataSize);
    cudaMemcpy(&d_data, &data, dataSize, cudaMemcpyHostToDevice);

    double *d_centroids = nullptr;
    int centroidSize = centroids.size() * centroids[0].size() * sizeof(double);
    cudaMalloc((void**)&d_centroids, centroidSize);
    cudaMemcpy(&d_centroids, &centroids, centroidSize, cudaMemcpyHostToDevice);

    double *d_flags = nullptr;
    int flagsSize = flags.size() * sizeof(int);
    cudaMalloc((void**)&d_flags, flagsSize);

}

std::vector<int> runCudaShmem(const matrix &data, matrix &centroids, std::vector<int> &flags, int maxIterations, double threshold)
{
    double *d_data = nullptr;
    int dataSize = data.size() * data[0].size() * sizeof(double);
    cudaMalloc((void **)&d_data, dataSize);
    cudaMemcpy(&d_data, &data, dataSize, cudaMemcpyHostToDevice);

    double *d_centroids = nullptr;
    int centroidSize = centroids.size() * centroids[0].size() * sizeof(double);
    cudaMalloc((void **)&d_centroids, centroidSize);
    cudaMemcpy(&d_centroids, &centroids, centroidSize, cudaMemcpyHostToDevice);

    double *d_flags = nullptr;
    int flagsSize = flags.size() * sizeof(int);
    cudaMalloc((void **)&d_flags, flagsSize);
}

std::vector<int> runThrust(const matrix &data, matrix &centroids, std::vector<int> &flags, int maxIterations, double threshold)
{
    double *d_data = nullptr;
    int dataSize = data.size() * data[0].size() * sizeof(double);
    cudaMalloc((void **)&d_data, dataSize);
    cudaMemcpy(&d_data, &data, dataSize, cudaMemcpyHostToDevice);

    double *d_centroids = nullptr;
    int centroidSize = centroids.size() * centroids[0].size() * sizeof(double);
    cudaMalloc((void **)&d_centroids, centroidSize);
    cudaMemcpy(&d_centroids, &centroids, centroidSize, cudaMemcpyHostToDevice);

    double *d_flags = nullptr;
    int flagsSize = flags.size() * sizeof(int);
    cudaMalloc((void **)&d_flags, flagsSize);
}
