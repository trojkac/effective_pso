#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "bbob_generators.cuh"

__constant__ double d_OMEGA = 0.64;
__constant__ double d_phi = 1.4;

__device__ double fitness_function(double x[], int number_of_variables)
{
    const double alpha = 100.0;
    size_t i = 0;
    double result;
    
    result = 0.0;
    for(i = 1; i < number_of_variables; ++i)
    {
        result += x[i] * x[i];
    }
    result = alpha * sqrt(result) + x[0] * x[0];

    return result;
}

__device__ double wrapped_fitness_function(double x[], int number_of_variables,
                                           double* xopt, double* M, double* b, double fopt)
{
    transform_vars_shift(x, number_of_variables, xopt);
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
                                 double* vars_shift_xopt,
                                 double* obj_shift_fopt)
    {
        bbob2009_compute_xopt(vars_shift_xopt, rseed, dimension);
        obj_shift_fopt[0] = bbob2009_compute_fopt(function, instance);

        double rot1[MAX_DIMENSIONS][MAX_DIMENSIONS];
        double rot2[MAX_DIMENSIONS][MAX_DIMENSIONS];

        bbob2009_compute_rotation(dimension, rot1, rseed + 1000000);
        bbob2009_compute_rotation(dimension, rot2, rseed);

        double *current_row;

        for(int i = 0; i < dimension; ++i)
        {
            vars_affine_b[i] = 0.0;
            current_row = vars_affine_m + i * dimension;
            for(int j = 0; j < dimension; ++j)
            {
                current_row[j] = 0.0;
                for(int k = 0; k < dimension; ++k)
                {
                    double exponent = 1.0 * (int)k / ((double)(long)dimension - 1.0);
                    current_row[j] += rot1[i][k] * pow(sqrt(10.0), exponent) * rot2[k][j];
                }
            }
        }
    }

    __global__ void transposeKernel(
        double* positions,
        double* velocities,
        double* personalBests,
        double* personalBestValues,
        int particlesCount,
        int dimensionsCount,
        double* xopt, double* M, double* b, double fopt)
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

        double tempLocation[MAX_DIMENSIONS];

        for(int i = 0; i < dimensionsCount; i++)
        {
            tempLocation[i] = particleLoc[i];
        }

        double newValue = wrapped_fitness_function(tempLocation, dimensionsCount, xopt, M, b, fopt);

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