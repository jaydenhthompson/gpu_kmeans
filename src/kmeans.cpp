#include "argparse.h"
#include "kmeans.h"
#include "seq_k_means.h"
#include "helpers.h"

#include <iostream>
#include <fstream>

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

    switch(opts.run_option)
    {
    case 0:
        break;
    case 1:
        break;
    case 2:
        break;
    case 3:
        break;
    }
}