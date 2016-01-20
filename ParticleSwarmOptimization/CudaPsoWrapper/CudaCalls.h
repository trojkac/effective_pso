#pragma once

#include <tuple>
#include <vector>

void runCuda(std::tuple<std::vector<double>, double>* local_endpoint, 
         std::tuple<std::vector<double>, double>* remote_endpoint,
         int iterations, int nr_of_particles, int dim, int sync_inveral);
