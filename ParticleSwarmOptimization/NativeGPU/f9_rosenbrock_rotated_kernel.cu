#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "bbob_generators.cuh"

__constant__ double d_OMEGA = 0.64;
__constant__ double d_phi = 1.4;

__constant__ double PI = 3.1415;

__device__ double fitness_function(double x[], int number_of_variables)
{
    size_t i = 0;
    double result;
    double s1 = 0.0, s2 = 0.0, tmp;

    for(i = 0; i < number_of_variables - 1; ++i)
    {
        tmp = (x[i] * x[i] - x[i + 1]);
        s1 += tmp * tmp;
        tmp = (x[i] - 1.0);
        s2 += tmp * tmp;
    }
    result = 100.0 * s1 + s2;

    return result;
}

__device__ double wrapped_fitness_function(double x[], int number_of_variables,
                                           double* M, double* b, double fopt)
{
    transform_vars_affine(x, number_of_variables, M, b);
    double temp[1];
    temp[0] = fitness_function(x, number_of_variables);
    transform_obj_shift(temp, 1, fopt);

    return temp[0];
}


extern "C" {
    __global__ void generateData(int dimension,
                                 int rseed,
                                 int function,
                                 int instance,
                                 double* vars_affine_m,
                                 double* vars_affine_b,
                                 double* obj_shift_fopt)
    {
        obj_shift_fopt[0] = bbob2009_compute_fopt(function, instance);

        double rot1[MAX_DIMENSIONS][MAX_DIMENSIONS];
        double rot2[MAX_DIMENSIONS][MAX_DIMENSIONS];

        bbob2009_compute_rotation(dimension, rot1, rseed + 1000000);
        bbob2009_compute_rotation(dimension, rot2, rseed);

        double *current_row;

        double factor = coco_double_max(1.0, sqrt((double)dimension) / 8.0);

        /* Compute affine transformation */
        for(int row = 0; row < dimension; ++row)
        {
            current_row = vars_affine_m + row * dimension;
            for(int column = 0; column < dimension; ++column)
            {
                current_row[column] = rot1[row][column];
                if(row == column)
                    current_row[column] *= factor;
            }
            vars_affine_b[row] = 0.5;
        }
    }

    __global__ void transposeKernel(
        double* positions,
        double* velocities,
        double* personalBests,
        double* personalBestValues,
        int particlesCount,
        int dimensionsCount,
        double* M, double* b, double fopt)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i >= particlesCount) return;

        double* particleLoc = positions + i * dimensionsCount;
        double* particleVel = velocities + i * dimensionsCount;

        for(int i = 0; i < dimensionsCount; i++)
        {
            particleLoc[i] += particleVel[i];
        }

        clamp(particleLoc, dimensionsCount, -5.0, 5.0);

        double newValue = wrapped_fitness_function(particleLoc, dimensionsCount, M, b, fopt);

        if(newValue < personalBestValues[i])
        {
            personalBestValues[i] = newValue;

            double* particlePersonalBest = personalBests + i * dimensionsCount;

            for(int i = 0; i < dimensionsCount; i++)
                particlePersonalBest[i] = particleLoc[i];
        }
    }

    __global__ void updateVelocityKernel(
        double* positions,
        double* velocities,
        double* personalBests,
        double* personalBestValues,
        int* neighbors,
        int particlesCount,
        int dimensionsCount,
        double phi1,
        double phi2)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i >= particlesCount) return;

        double* particleLoc = positions + i * dimensionsCount;
        double* particleVel = velocities + i * dimensionsCount;
        double* particleBest = personalBests + i * dimensionsCount;
        double particleBestValue = personalBestValues[i];

        int* particleNeighbors = neighbors + i * 2;

        int leftNeighborId = particleNeighbors[0];
        double* leftNeighborBest = personalBests + leftNeighborId * dimensionsCount;
        double leftNeighborBestVal = personalBestValues[leftNeighborId];

        int rightNeighborId = particleNeighbors[1];
        double* rightNeighborBest = personalBests + rightNeighborId * dimensionsCount;
        double rightNeighborBestVal = personalBestValues[rightNeighborId];

        double* globalBest = particleBest;
        double globalBestVal = particleBestValue;

        if(leftNeighborBestVal < globalBestVal)
        {
            globalBest = leftNeighborBest;
            globalBestVal = leftNeighborBestVal;
        }

        if(rightNeighborBestVal < globalBestVal)
        {
            globalBest = rightNeighborBest;
        }

        double toPersonalBest[MAX_DIMENSIONS];
        vector_between(particleLoc, particleBest, dimensionsCount, toPersonalBest);

        double toGlobalBest[MAX_DIMENSIONS];
        vector_between(particleLoc, globalBest, dimensionsCount, toGlobalBest);

        for(int i = 0; i < dimensionsCount; i++)
        {
            particleVel[i] = particleVel[i] * d_OMEGA + phi1 * toGlobalBest[i] + phi2 * toPersonalBest[i];
        }
    }
}