#pragma once

#include <getopt.h>
#include <iostream>
#include <stdlib.h>
#include <string>

struct options_t {
    std::string in_file;
    int num_cluster;
    int dims;
    int max_num_iter;
    double convergence_threshold;
    bool output_centroids;
    int run_option;
    uint seed;
};

void get_opts(int argc,
              char **argv,
              options_t &opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t-i <file_path>" << std::endl;
        std::cout << "\t-k <num_clusters>" << std::endl;
        std::cout << "\t-d <dims>" << std::endl;
        std::cout << "\t-m <max_num_iters>" << std::endl;
        std::cout << "\t-t <threshold>" << std::endl;
        std::cout << "\t-c <output_centroids>" << std::endl;
        std::cout << "\t-s <seed>" << std::endl;
        std::cout << "\t-r <sequential>" << std::endl;
        exit(0);
    }

    opts.output_centroids = false;

    struct option l_opts[] = {
        {"in", required_argument, NULL, 'i'},
        {"num_cluster", required_argument, NULL, 'k'},
        {"dims", required_argument, NULL, 'd'},
        {"max_num_iter", required_argument, NULL, 'm'},
        {"threshold", required_argument, NULL, 't'},
        {"seed", required_argument, NULL, 's'},
        {"run_option", required_argument, NULL, 'r'},
        {"output_centroids", no_argument, NULL, 'c'}
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "i:k:d:m:t:r:s:c", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'i':
            opts.in_file = std::string(optarg);
            break;
        case 'k':
            opts.num_cluster = atoi((char *)optarg);
            break;
        case 'd':
            opts.dims = atoi((char *)optarg);
            break;
        case 'm':
            opts.max_num_iter = atoi((char *)optarg);
            break;
        case 't':
            opts.convergence_threshold = std::stod(std::string(optarg));
            break;
        case 'c':
            opts.output_centroids = true;
            break;
        case 'r':
            opts.run_option = atoi((char*)optarg);
            break;
        case 's':
            opts.seed = (uint)atoi((char*)optarg);
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}