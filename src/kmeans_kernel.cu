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

__global__ void cudaCalculateFlags(double *d_data, double *d_centroids, int *d_flags, int d_dataSize, int d_numCentroids, int d_dimensions)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(index >= d_dataSize) return;
    if(index == 0) printf("data 0 %lf", d_data[0]);

    int assigned = 0;
    double minDist = INFINITY;
    for (int i = 0; i < d_numCentroids; i++)
    {
        double dist = euclideanDistance(&d_data[index], &d_centroids[i], d_dimensions);
        if(dist < minDist)
        {
            minDist = dist;
            assigned = i;
        }
    }
    d_flags[index] = assigned;
}

__global__ void cudaCalculateNewCentroids(double *d_data, double *d_centroids, int *d_flags, int *d_numAssigned, int d_dataSize, int d_numCentroids, int d_dimensions)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("index: %d\n", index);

    if(index >= d_dataSize) return;

    int assignedCentroid = d_flags[index];

    for(int i = 0; i < d_dimensions; i++)
    {
        atomicAdd(&d_centroids[assignedCentroid*d_dimensions + i], d_data[index*d_dimensions + i]);
        atomicAdd(&d_numAssigned[assignedCentroid], 1);
    }

    __syncthreads();

    if(index < d_numCentroids)
    {
        d_centroids[index] /= d_numAssigned[index];
    }
    
    __syncthreads();
}


std::vector<float> runCudaBasic(const matrix &data, matrix &centroids, std::vector<int> &flags, int maxIterations, double threshold)
{
    matrix newCentroids(centroids.size(), std::vector<double>(centroids[0].size()));
    std::vector<float> times;

    double *d_data = nullptr;
    int dataSize = data.size() * data[0].size() * sizeof(double);
    auto err = cudaMalloc((void**)&d_data, dataSize);
    if(err != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(d_data, data.data(), dataSize, cudaMemcpyHostToDevice);

    double *d_centroids = nullptr;
    int centroidSize = centroids.size() * centroids[0].size() * sizeof(double);
    cudaMalloc((void**)&d_centroids, centroidSize);
    cudaMemcpy(d_centroids, centroids.data(), centroidSize, cudaMemcpyHostToDevice);

    double *d_newCentroids = nullptr;
    cudaMalloc((void**)&d_newCentroids, centroidSize);
    cudaMemset(d_newCentroids, 0, centroidSize);

    int *d_flags = nullptr;
    int flagsSize = flags.size() * sizeof(int);
    cudaMalloc((void**)&d_flags, flagsSize);

    int *d_numAssigned = nullptr;
    cudaMalloc((void**)&d_numAssigned, centroids.size() * sizeof(int));
    cudaMemset(d_numAssigned, 0, centroids.size() * sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (data.size() + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < maxIterations; i++)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        cudaCalculateFlags<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_centroids, d_flags, data.size(), centroids.size(), data[0].size());
        cudaDeviceSynchronize();
        cudaCalculateNewCentroids<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_newCentroids, d_flags, d_numAssigned, data.size(), centroids.size(), data[0].size());
        cudaDeviceSynchronize();

        cudaMemcpy(newCentroids.data(), d_newCentroids, centroidSize, cudaMemcpyDeviceToHost);
        std::cout << "printing new centroids\n";
        for (auto & e : newCentroids){
            for (auto & f : e) {
                std::cout << f << " ";
            }
            std::cout << std::endl;
        }
        auto movement = calculateMovement(centroids, newCentroids);
        if(movement <= threshold)
        {
            break;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time = 0;
        cudaEventElapsedTime(&time, start, stop);
        times.push_back(time);
        
        centroids = newCentroids;
        cudaMemset(d_newCentroids, 0, centroidSize);
    }

    cudaMemcpy(flags.data(), d_flags, flagsSize, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_newCentroids);
    cudaFree(d_flags);
    return times;
}

std::vector<float> runCudaShmem(const matrix &data, matrix &centroids, std::vector<int> &flags, int maxIterations, double threshold)
{
    std::vector<float>times;
    return times;
}

std::vector<float> runThrust(const matrix &data, matrix &centroids, std::vector<int> &flags, int maxIterations, double threshold)
{
    std::vector<float>times;
    return times;
}