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

int main(int argc, char **argv)
{
    options_t opts;
    get_opts(argc, argv, opts);

    int numRows;
    auto input = parseFile(opts.in_file, opts.dims, numRows);
    auto centers = getRandomCentroids(input, opts.dims, opts.num_cluster, opts.seed);
    std::vector<int> flags(numRows, -1);

    std::vector<int> iterations;
    switch(opts.run_option)
    {
    case 0:
        iterations = runSequentialKMeans(input, centers, flags, opts.max_num_iter, opts.convergence_threshold);
        break;
    case 1:
        break;
    case 2:
        break;
    case 3:
        break;
    }

    std::cout << iterations.size() << ","
              << std::reduce(iterations.begin(), iterations.end()) / static_cast<double>(iterations.size())
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
    }

    return 0;
}