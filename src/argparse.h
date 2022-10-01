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

void get_opts(int argc, char **argv, options_t &opts);