#include "helpers.h"
#include "kmeans.h"

// CUDA Includes
#include <cuda_runtime.h>

// thrust includes
#include <thrust/device_vector.h>

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

__global__ void cudaAddNewCentroids(double *d_data, double *d_centroids, int *d_flags, int *d_numAssigned, int d_dataSize, int d_numCentroids, int d_dimensions)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= d_dataSize) return;

    int assignedCentroid = d_flags[index];

    for(int i = 0; i < d_dimensions; i++)
    {
        atomicAdd(&d_centroids[assignedCentroid*d_dimensions + i], d_data[index*d_dimensions + i]);
    }
}

__global__ void cudaAverageNewCentroids(double *d_data, double *d_centroids, int *d_flags, int *d_numAssigned, int d_dataSize, int d_numCentroids, int d_dimensions)
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

__global__ void shmemCalculateFlags(double *d_data, double *d_centroids, int *d_flags, int *d_numAssigned, int d_dataSize, int d_numCentroids, int d_dimensions)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int sharedIndex = threadIdx.x;

    if (index >= d_dataSize)
        return;

    extern __shared__ int s_numAssigned[];
    if (sharedIndex == 0)
    {
        for (int i = 0; i < d_numCentroids; i++)
        {
            s_numAssigned[i] = 0;
        }
    }
    __syncthreads();

    int assigned = 0;
    double minDist = INFINITY;
    for (int i = 0; i < d_numCentroids; i++)
    {
        double dist = euclideanDistance(&d_data[index * d_dimensions], &d_centroids[i * d_dimensions], d_dimensions);
        if (dist < minDist)
        {
            minDist = dist;
            assigned = i;
        }
    }
    d_flags[index] = assigned;
    atomicAdd(&s_numAssigned[assigned], 1);

    __syncthreads();
    if (sharedIndex == 0)
    {
        for (int i = 0; i < d_numCentroids; i++)
        {
            atomicAdd(&d_numAssigned[i], s_numAssigned[i]);
        }
    }
}

__global__ void shmemAddNewCentroids(double *d_data, double *d_centroids, int *d_flags, int *d_numAssigned, int d_dataSize, int d_numCentroids, int d_dimensions)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= d_dataSize) return;

    int assignedCentroid = d_flags[index];

    for(int i = 0; i < d_dimensions; i++)
    {
        atomicAdd(&d_centroids[assignedCentroid*d_dimensions + i], d_data[index*d_dimensions + i]);
    }
}

__global__ void shmemAverageNewCentroids(double *d_data, double *d_centroids, int *d_flags, int *d_numAssigned, int d_dataSize, int d_numCentroids, int d_dimensions)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < d_numCentroids)
    {
        for (int i = 0; i < d_dimensions; i++)
        {
            d_centroids[index * d_dimensions + i] /= d_numAssigned[index];
        }
    }
}

std::vector<float> runCuda(const std::vector<double> &data, std::vector<double> &centroids, std::vector<int> &flags, int option, int dimensions, int numData, int numClusters, int maxIterations, double threshold)
{
    ////////////////////
    // Host variables //
    ////////////////////

    std::vector<double> newCentroids(centroids.size(), 0);
    std::vector<float> times;

    ////////////////////
    // CUDA variables //
    ////////////////////

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

    ////////////////////
    // Execution Loop //
    ////////////////////

    int threadsPerBlock = 256;
    int blocksPerGrid = (numData + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < maxIterations; i++)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        /////////////////
        // Run Kernels //
        /////////////////

        cudaMemset(d_numAssigned, 0, numClusters * sizeof(int));
        switch(option)
        {
        case 1:
            cudaCalculateFlags<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_newCentroids, d_flags, d_numAssigned, numData, numClusters, dimensions);
            cudaDeviceSynchronize();
            cudaMemset(d_newCentroids, 0, centroidSize);
            cudaAddNewCentroids<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_newCentroids, d_flags, d_numAssigned, numData, numClusters, dimensions);
            cudaDeviceSynchronize();
            cudaAverageNewCentroids<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_newCentroids, d_flags, d_numAssigned, numData, numClusters, dimensions);
            cudaDeviceSynchronize();
            break;
        case 2:
            shmemCalculateFlags<<<blocksPerGrid, threadsPerBlock, numClusters*sizeof(int)>>>(d_data, d_newCentroids, d_flags, d_numAssigned, numData, numClusters, dimensions);
            cudaDeviceSynchronize();
            cudaMemset(d_newCentroids, 0, centroidSize);
            shmemAddNewCentroids<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_newCentroids, d_flags, d_numAssigned, numData, numClusters, dimensions);
            cudaDeviceSynchronize();
            shmemAverageNewCentroids<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_newCentroids, d_flags, d_numAssigned, numData, numClusters, dimensions);
            cudaDeviceSynchronize();
            break;
        default:
            std::cout << "Wrong option for -r" << std::endl;
            return times;
            break;
        }

        /////////////////////////
        // Calculate Threshold //
        /////////////////////////

        cudaMemcpy(&newCentroids[0], d_newCentroids, centroidSize, cudaMemcpyDeviceToHost);
        auto movement = calculateVectorMovement(centroids, newCentroids, numClusters, dimensions);
        /////////////////
        // Record Time //
        /////////////////

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time = 0;
        cudaEventElapsedTime(&time, start, stop);
        times.push_back(time);

        // record new centroids
        centroids = newCentroids;
        if(movement <= threshold)
        {
            break;
        }
    }

    // record flags
    cudaMemcpy(flags.data(), d_flags, flagsSize, cudaMemcpyDeviceToHost);

    // memory management
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_newCentroids);
    cudaFree(d_flags);

    return times;
}

std::vector<float> runThrust(const std::vector<double> &data, std::vector<double> &centroids, std::vector<int> &flags, int option, int dimensions, int numData, int numClusters, int maxIterations, double threshold)
{
    std::vector<double> newCentroids(centroids.size(), 0);
    std::vector<float> times;

    thrust::device_vector<double> d_data(data);
    thrust::device_vector<double> d_centroids(centroids);
    thrust::device_vector<double> d_newCentroids(centroids);
    thrust::device_vector<int> d_flags(flags);
    thrust::device_vector<int> d_numAssigned(centroids.size());

    thrust::device_vector<int> originalCentroidOrder(centroids.size() * dimensions);
    thrust::sequence(originalCentroidOrder.begin(), originalCentroidOrder.end());

    thrust::device_vector<int> originalDataOrder(data.size() * dimensions);
    thrust::sequence(originalDataOrder.begin(), originalDataOrder.end());


    ////////////////////
    // Execution Loop //
    ////////////////////

    int threadsPerBlock = 256;
    int blocksPerGrid = (numData + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < maxIterations; i++)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        /////////////////
        // Run Kernels //
        /////////////////

        thrust::fill(d_numAssigned.begin(), d_numAssigned.end(), 0);
        cudaCalculateFlags<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(d_data.data()), thrust::raw_pointer_cast(d_newCentroids.data()), thrust::raw_pointer_cast(d_flags.data()), thrust::raw_pointer_cast(d_numAssigned.data()), numData, numClusters, dimensions);
        cudaDeviceSynchronize();
        thrust::fill(d_newCentroids.begin(), d_newCentroids.end(), 0);
        cudaAddNewCentroids<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(d_data.data()), thrust::raw_pointer_cast(d_newCentroids.data()), thrust::raw_pointer_cast(d_flags.data()), thrust::raw_pointer_cast(d_numAssigned.data()), numData, numClusters, dimensions);
        cudaDeviceSynchronize();
        cudaAverageNewCentroids<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(d_data.data()), thrust::raw_pointer_cast(d_newCentroids.data()), thrust::raw_pointer_cast(d_flags.data()), thrust::raw_pointer_cast(d_numAssigned.data()), numData, numClusters, dimensions);
        cudaDeviceSynchronize();

        /////////////////////////
        // Calculate Threshold //
        /////////////////////////

        thrust::copy(d_newCentroids.begin(), d_newCentroids.end(), newCentroids.begin());
        auto movement = calculateVectorMovement(centroids, newCentroids, numClusters, dimensions);
        /////////////////
        // Record Time //
        /////////////////

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time = 0;
        cudaEventElapsedTime(&time, start, stop);
        times.push_back(time);

        // record new centroids
        centroids = newCentroids;
        if(movement <= threshold)
        {
            break;
        }
    }

    return times;
}