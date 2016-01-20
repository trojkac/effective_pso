#pragma once

#include "stdafx.h"

#include "CudaCalls.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <tuple>
#include <vector>

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

__global__ void initialize_particles(double** d_particles, double** d_velocities, double** d_personal_bests_locations,
                                     double* d_personal_bests_values, int dim, int nr_of_particles,
                                     double** d_neighbor_bests_locations, double* d_neighbor_bests_values,
                                     double** d_proxy_best_locations, double* d_proxy_best_values)
{
    int particle_nr = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < dim; i++)
    {
        d_particles[particle_nr][i] = 0.0;
        d_velocities[particle_nr][i] = 1.0;
        d_personal_bests_locations[particle_nr][i] = 0.0;
        d_personal_bests_values[particle_nr] = INT_MAX;
        d_neighbor_bests_locations[particle_nr][i] = 0.0;
        d_neighbor_bests_values[particle_nr] = INT_MAX;
        d_proxy_best_locations[particle_nr][i] = 0.0;
        d_proxy_best_values[particle_nr] = INT_MAX;
    }
}

__global__ void compute_function(double** d_particles, double** d_velocities, double** d_personal_bests_locations,
                                 double* d_personal_bests_values, int dim, int nr_of_particles,
                                 double** d_neighbor_bests_locations, double* d_neighbor_bests_values)
{
    int particle_nr = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0.0;
    for(int i = 0; i < dim; i++)
    {
        value += d_particles[particle_nr][i] * d_particles[particle_nr][i];
    }

    if(value < d_personal_bests_values[particle_nr])
    {
        d_personal_bests_values[particle_nr] = value;
        d_personal_bests_locations[particle_nr] = d_particles[particle_nr];
    }
}

__global__ void actualize_bests(double** d_particles, double** d_velocities, double** d_personal_bests_locations,
                                double* d_personal_bests_values, int dim, int nr_of_particles,
                                double** d_neighbor_bests_locations, double* d_neighbor_bests_values,
                                double** d_proxy_best_locations, double* d_proxy_best_values)
{
    int particle_nr = blockIdx.x * blockDim.x + threadIdx.x;
    int first_particle_in_block = blockIdx.x * blockDim.x;

    double* new_best_neighbor_location = new double[dim], new_best_neighbor_value = INT_MAX;

    for(int i = first_particle_in_block; i < first_particle_in_block + blockDim.x; i++)
    {
        if(d_neighbor_bests_values[particle_nr] > d_personal_bests_values[i])
        {
            new_best_neighbor_location = d_personal_bests_locations[i];
            new_best_neighbor_value = d_personal_bests_values[i];
        }
    }

    if(new_best_neighbor_value > d_proxy_best_values[particle_nr])
    {
        new_best_neighbor_location = d_proxy_best_locations[particle_nr];
        new_best_neighbor_value = d_proxy_best_values[particle_nr];
    }

    if(new_best_neighbor_value < INT_MAX)
    {
        d_neighbor_bests_values[particle_nr] = new_best_neighbor_value;

        for(int i = 0; i < dim; i++)
        {
            d_neighbor_bests_locations[particle_nr][i] = new_best_neighbor_location[i];
        }
    }
}

__global__ void compute_velocities(double** d_particles, double** d_velocities, double** d_personal_bests_locations,
                                   double* d_personal_bests_values, int dim, int nr_of_particles,
                                   double** d_neighbor_bests_locations, double* d_neighbor_bests_values)
{
    int particle_nr = blockIdx.x * blockDim.x + threadIdx.x;

    double r1 = 0.5;
    double r2 = 0.5;

    for(int i = 0; i < dim; i++)
    {
        d_velocities[particle_nr][i] += r1 * (d_personal_bests_locations[particle_nr][i] - d_particles[particle_nr][i]) +
            r2 * (d_neighbor_bests_locations[particle_nr][i] - d_particles[particle_nr][i]);
    }
}

__global__ void actualize_locations(double** d_particles, double** d_velocities, double** d_personal_bests_locations,
                                    double* d_personal_bests_values, int dim, int nr_of_particles,
                                    double** d_neighbor_bests_locations, double* d_neighbor_bests_values)
{
    int particle_nr = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = 0; i < dim; i++)
    {
        d_particles[particle_nr][i] += d_velocities[particle_nr][i];
    }
}

void runCuda(std::tuple<std::vector<double>, double>* local_endpoint, std::tuple<std::vector<double>, double>* remote_endpoint,
             int iterations, int nr_of_particles, int dim, int sync_interval)
{
    int particle_size = sizeof(double);

    int sync_index = 0;

    dim3 threads_per_block(16);
    dim3 num_of_blocks(nr_of_particles / (threads_per_block.x * threads_per_block.y * threads_per_block.z));

    double **h_particles, **h_personal_bests_locations, *h_personal_bests_values, **h_velocities,
        **h_neighbor_bests_locations, *h_neighbor_bests_values, **h_proxy_best_locations, *h_proxy_best_values;

    //alokuj pamiêæ na CPU
    h_particles = new double*[nr_of_particles];
    for(int i = 0; i < nr_of_particles; i++)
    {
        h_particles[i] = new double[dim];
    }

    h_personal_bests_locations = new double*[nr_of_particles];
    for(int i = 0; i < nr_of_particles; i++)
    {
        h_personal_bests_locations[i] = new double[dim];
    }

    h_personal_bests_values = new double[nr_of_particles];

    h_velocities = new double*[nr_of_particles];
    for(int i = 0; i < nr_of_particles; i++)
    {
        h_velocities[i] = new double[dim];
    }

    h_neighbor_bests_locations = new double*[nr_of_particles];
    for(int i = 0; i < nr_of_particles; i++)
    {
        h_neighbor_bests_locations[i] = new double[dim];
    }

    h_neighbor_bests_values = new double[nr_of_particles];

    h_proxy_best_locations = new double*[nr_of_particles];
    for(int i = 0; i < nr_of_particles; i++)
    {
        h_proxy_best_locations[i] = new double[dim];
    }

    h_proxy_best_values = new double[nr_of_particles];

    double **d_particles, **d_personal_bests_locations, *d_personal_bests_values, **d_velocities,
        **d_neighbor_bests_locations, *d_neighbor_bests_values, **d_proxy_best_locations, *d_proxy_best_values;

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc(&d_particles, nr_of_particles * particle_size * dim);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&d_personal_bests_locations, nr_of_particles*particle_size * dim);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&d_personal_bests_values, nr_of_particles * particle_size);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&d_velocities, nr_of_particles * particle_size * dim);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&d_neighbor_bests_locations, nr_of_particles * particle_size * dim);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&d_neighbor_bests_values, nr_of_particles * particle_size);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&d_proxy_best_locations, nr_of_particles * particle_size * dim);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&d_proxy_best_values, nr_of_particles * particle_size);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    //inicjalizuj po³o¿enia, prêdkoœci oraz bests
    initialize_particles KERNEL_ARGS2(num_of_blocks, threads_per_block) (d_particles, d_velocities, d_personal_bests_locations, d_personal_bests_values, dim, nr_of_particles, d_neighbor_bests_locations, d_neighbor_bests_values, d_proxy_best_locations, d_proxy_best_values);

    for(int i = 1; i <= iterations; i++)
    {
        compute_function KERNEL_ARGS2(num_of_blocks, threads_per_block) (d_particles, d_velocities, d_personal_bests_locations, d_personal_bests_values, dim, nr_of_particles, d_neighbor_bests_locations, d_neighbor_bests_values);
        actualize_bests KERNEL_ARGS2(num_of_blocks, threads_per_block) (d_particles, d_velocities, d_personal_bests_locations, d_personal_bests_values, dim, nr_of_particles, d_neighbor_bests_locations, d_neighbor_bests_values, d_proxy_best_locations, d_proxy_best_values);
        compute_velocities KERNEL_ARGS2(num_of_blocks, threads_per_block) (d_particles, d_velocities, d_personal_bests_locations, d_personal_bests_values, dim, nr_of_particles, d_neighbor_bests_locations, d_neighbor_bests_values);
        actualize_locations KERNEL_ARGS2(num_of_blocks, threads_per_block) (d_particles, d_velocities, d_personal_bests_locations, d_personal_bests_values, dim, nr_of_particles, d_neighbor_bests_locations, d_neighbor_bests_values);

        if(local_endpoint && (i % sync_interval == 0))
        {
            auto remote_best = *remote_endpoint;
            auto personal_best = *local_endpoint;

            if(remote_endpoint && local_endpoint && std::get<1>(*local_endpoint) > std::get<1>(*remote_endpoint))
            {
                *local_endpoint = *remote_endpoint;
            }

            cudaStatus = cudaMemcpy(h_personal_bests_locations, d_personal_bests_locations, nr_of_particles * particle_size * dim,
                                    cudaMemcpyDeviceToHost);
            if(cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }

            cudaStatus = cudaMemcpy(h_personal_bests_values, d_personal_bests_values, nr_of_particles * particle_size,
                                    cudaMemcpyDeviceToHost);
            if(cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }

            cudaStatus = cudaMemcpy(h_proxy_best_locations, d_proxy_best_locations, nr_of_particles * particle_size * dim,
                                    cudaMemcpyDeviceToHost);
            if(cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }

            cudaStatus = cudaMemcpy(h_proxy_best_values, d_proxy_best_values, nr_of_particles * particle_size,
                                    cudaMemcpyDeviceToHost);
            if(cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }

            if(std::get<1>(*local_endpoint) > h_personal_bests_values[sync_index])
            {
                auto loc = std::vector<double>(h_personal_bests_locations[sync_index], h_personal_bests_locations[sync_index] + dim);
                local_endpoint = new std::tuple<std::vector<double>, double>(loc, h_personal_bests_values[sync_index]);
            }        

            auto temp = std::get<0>(*local_endpoint);

            std::copy(temp.begin(), temp.end(), h_proxy_best_locations[sync_index]);
            h_proxy_best_values[sync_index] = std::get<1>(*local_endpoint);

            cudaStatus = cudaMemcpy(d_proxy_best_locations, h_proxy_best_locations, nr_of_particles * particle_size * dim,
                                    cudaMemcpyHostToDevice);
            if(cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }

            cudaStatus = cudaMemcpy(d_proxy_best_values, h_proxy_best_values, nr_of_particles * particle_size,
                                    cudaMemcpyHostToDevice);
            if(cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }
        }
    }

    cudaStatus = cudaMemcpy(h_particles, d_particles, nr_of_particles * particle_size * dim, cudaMemcpyDeviceToHost);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(h_personal_bests_locations, d_personal_bests_locations, nr_of_particles * particle_size * dim,
                            cudaMemcpyDeviceToHost);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(h_personal_bests_values, d_personal_bests_values, nr_of_particles * particle_size,
                            cudaMemcpyDeviceToHost);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(h_neighbor_bests_locations, d_neighbor_bests_locations, nr_of_particles * particle_size * dim,
                            cudaMemcpyDeviceToHost);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(h_neighbor_bests_values, d_neighbor_bests_values, nr_of_particles * particle_size,
                            cudaMemcpyDeviceToHost);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        goto Error;
    }

Error:
    cudaFree(d_particles);
    cudaFree(d_neighbor_bests_locations);
    cudaFree(d_personal_bests_locations);
    cudaFree(d_velocities);
    cudaFree(d_neighbor_bests_values);
    cudaFree(d_personal_bests_values);
}
