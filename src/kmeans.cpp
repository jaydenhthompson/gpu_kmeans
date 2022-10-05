#include "argparse.h"
#include "kmeans.h"
#include "seq_k_means.h"
#include "helpers.h"

#include <iostream>
#include <fstream>
#include <numeric>

matrix parseFile(std::string file, int dimensions, int &numRows)
{
    std::ifstream inFile(file);
    inFile >> numRows;

    int rownum;
    auto input = matrix(numRows, std::vector<double>(dimensions));
    for(int i = 0; i < numRows; i++)
    {
        inFile >> rownum;
        for(int j = 0; j < dimensions; j++)
        {
            inFile >> input[i][j];
        }
    }
    return input;
}

double avgVector(const std::vector<float>&v)
{
    double avg = 0;
    for(auto & e : v)
    {
        avg += e;
    }
    return avg / static_cast<double>(v.size());
}

std::vector<double> convertMatrix(const matrix &m)
{
    std::vector<double> v;
    for (auto &e : m)
    {
        for (auto &f : e)
        {
            v.push_back(f);
        }
    }
    return v;
}

void convertVector(const std::vector<double> v, matrix& m)
{
    for(int i = 0; i < m.size(); i++)
    {
        for(int j = 0; j < m[0].size(); j++)
        {
            m[i][j] = v[i*m[0].size() + j];
        }
    }
}

int main(int argc, char **argv)
{
    options_t opts;
    get_opts(argc, argv, opts);

    int numRows;
    auto input = parseFile(opts.in_file, opts.dims, numRows);
    auto centers = getRandomCentroids(input, opts.dims, opts.num_cluster, opts.seed);
    std::vector<int> flags(numRows, -1);

    auto dataVector = convertMatrix(input);
    auto centerVector = convertMatrix(centers);

    std::vector<float> iterations;
    if (opts.run_option == 0)
    {
        iterations = runSequentialKMeans(input, centers, flags, opts.max_num_iter, opts.convergence_threshold);
    }
    else if(opts.run_option == 3)
    {
        iterations = runThrust(dataVector, centerVector, flags, opts.run_option, opts.dims, numRows, opts.num_cluster, opts.max_num_iter, opts.convergence_threshold);
        convertVector(centerVector, centers);
    }
    else
    {
        iterations = runCuda(dataVector, centerVector, flags, opts.run_option, opts.dims, numRows, opts.num_cluster, opts.max_num_iter, opts.convergence_threshold);
        convertVector(centerVector, centers);
    }

    std::cout << iterations.size() << ","
              << avgVector(iterations)
              << std::endl;

    if(opts.output_centroids) 
    {
        for(int i = 0; i < opts.num_cluster; i++)
        {
            std::cout << i << " ";
            for(auto & e : centers[i])
            {
                std::cout << e << " ";
            }
            std::cout << std::endl;
        }
    }
    else
    {
        std::cout << "clusters:";
        for(auto &e : flags)
        {
            std::cout << " " << e;
        }
        std::cout << std::endl;
    }

    return 0;
}