#include "helpers.h"
#include "kmeans.h"

// CUDA Includes
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

__device__ double euclideanDistance(double *a, double *b, int dim)
{
    double dist = 0.0;
    for(int i = 0; i < dim; i++)
    {
        dist += pow(a[i] - b[i], 2);
    }
    return sqrt(dist);
}

__global__ void cudaCalculateFlags(double *d_data, double *d_centroids, int *d_flags, int *d_numAssigned, int d_dataSize, int d_numCentroids, int d_dimensions)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(index >= d_dataSize) return;

    int assigned = 0;
    double minDist = INFINITY;
    for (int i = 0; i < d_numCentroids; i++)
    {
        double dist = euclideanDistance(&d_data[index*d_dimensions], &d_centroids[i*d_dimensions], d_dimensions);
        if(dist < minDist)
        {
            minDist = dist;
            assigned = i;
        }
    }
    atomicAdd(&d_numAssigned[assigned], 1);
    d_flags[index] = assigned;
}

__global__ void cudaCalculateNewCentroids(double *d_data, double *d_centroids, int *d_flags, int *d_numAssigned, int d_dataSize, int d_numCentroids, int d_dimensions)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= d_dataSize) return;

    int assignedCentroid = d_flags[index];

    for(int i = 0; i < d_dimensions; i++)
    {
        atomicAdd(&d_centroids[assignedCentroid*d_dimensions + i], d_data[index*d_dimensions + i]);
    }
}

__global__ void averageNewCentroids(double *d_data, double *d_centroids, int *d_flags, int *d_numAssigned, int d_dataSize, int d_numCentroids, int d_dimensions)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= d_dataSize) return;

    if(index < d_numCentroids)
    {
        for(int i = 0; i < d_dimensions; i++)
        {
            d_centroids[index*d_dimensions + i] /= d_numAssigned[index];
        }
    }
}

std::vector<float> runCudaBasic(const std::vector<double> &data, std::vector<double> &centroids, std::vector<int> &flags, int dimensions, int numData, int numClusters, int maxIterations, double threshold)
{
    std::vector<double> newCentroids(centroids.size(), 0);
    std::vector<float> times;

    double *d_data = nullptr;
    int dataSize = data.size() * sizeof(double);
    cudaMalloc((void**)&d_data, dataSize);
    cudaMemcpy(d_data, &data[0], dataSize, cudaMemcpyHostToDevice);

    double *d_centroids = nullptr;
    int centroidSize = centroids.size() * sizeof(double);
    cudaMalloc((void**)&d_centroids, centroidSize);
    cudaMemcpy(d_centroids, &centroids[0], centroidSize, cudaMemcpyHostToDevice);

    double *d_newCentroids = nullptr;
    cudaMalloc((void**)&d_newCentroids, centroidSize);
    cudaMemcpy(d_newCentroids, &centroids[0], centroidSize, cudaMemcpyHostToDevice);

    int *d_flags = nullptr;
    int flagsSize = flags.size() * sizeof(int);
    cudaMalloc((void**)&d_flags, flagsSize);

    int *d_numAssigned = nullptr;
    cudaMalloc((void**)&d_numAssigned, numClusters * sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numData + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < maxIterations; i++)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        cudaMemset(d_numAssigned, 0, numClusters * sizeof(int));
        cudaCalculateFlags<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_newCentroids, d_flags, d_numAssigned, numData, numClusters, dimensions);
        cudaMemset(d_newCentroids, 0, centroidSize);
        cudaDeviceSynchronize();
        cudaCalculateNewCentroids<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_newCentroids, d_flags, d_numAssigned, numData, numClusters, dimensions);
        cudaDeviceSynchronize();
        averageNewCentroids<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_newCentroids, d_flags, d_numAssigned, numData, numClusters, dimensions);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaMemcpy(&newCentroids[0], d_newCentroids, centroidSize, cudaMemcpyDeviceToHost);

        auto movement = calculateVectorMovement(centroids, newCentroids, numClusters, dimensions);
        if(movement <= threshold)
        {
            break;
        }

        float time = 0;
        cudaEventElapsedTime(&time, start, stop);
        times.push_back(time);

        
        centroids = newCentroids;
    }

    cudaMemcpy(flags.data(), d_flags, flagsSize, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_newCentroids);
    cudaFree(d_flags);
    return times;
}

std::vector<float> runCudaShmem(const std::vector<double> &data, std::vector<double> &centroids, std::vector<int> &flags, int maxIterations, double threshold)
{
    std::vector<float>times;
    return times;
}

std::vector<float> runThrust(const std::vector<double> &data, std::vector<double> &centroids, std::vector<int> &flags, int maxIterations, double threshold)
{
    std::vector<float>times;
    return times;
}
