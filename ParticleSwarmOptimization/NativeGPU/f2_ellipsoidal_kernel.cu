#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "bbob_generators.cuh"

__constant__ double d_OMEGA = 0.64;
__constant__ double d_phi = 1.4;

__constant__ double PI = 3.1415;

__device__ double fitness_function(double x[], int number_of_variables)
{
    const double condition = 1.0e6;
    size_t i = 0;
    double result;

    result = x[i] * x[i];
    for(i = 1; i < number_of_variables; ++i)
    {
        const double exponent = 1.0 * (double)(long)i / ((double)(long)number_of_variables - 1.0);
        result += pow(condition, exponent) * x[i] * x[i];
    }

    return result;
}

__device__ double wrapped_fitness_function(double x[], int number_of_variables,
                                           double* xopt, double fopt)
{
    transform_vars_shift(x, number_of_variables, xopt);
    transform_vars_oscillate(x, number_of_variables);
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
                                 double* vars_shift_xopt,
                                 double* obj_shift_fopt)
    {
        bbob2009_compute_xopt(vars_shift_xopt, rseed, dimension);
        obj_shift_fopt[0] = bbob2009_compute_fopt(function, instance);
    }

    __global__ void kernelUpdateParticle(double *positions, double *velocities,
                                         double *pBests, double *gBest,
                                         int particlesCount, int dimensionsCount,
                                         double r1, double r2)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i >= particlesCount * dimensionsCount)
            return;

        velocities[i] = d_OMEGA * velocities[i] + r1 * (pBests[i] - positions[i])
            + r2 * (gBest[i % dimensionsCount] - positions[i]);

        // Update posisi particle
        positions[i] += velocities[i];
    }

    __global__ void kernelUpdatePBest(double *positions, double *pBests, double* gBest,
                                      int particlesCount, int dimensionsCount,
                                      double* xopt, double fopt)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        double tempParticle1[MAX_DIMENSIONS];
        double tempParticle2[MAX_DIMENSIONS];

        if(i >= particlesCount * dimensionsCount || i % dimensionsCount != 0)
            return;

        for(int j = 0; j < dimensionsCount; j++)
        {
            tempParticle1[j] = positions[i + j];
            tempParticle2[j] = pBests[i + j];
        }

        if(wrapped_fitness_function(tempParticle1, dimensionsCount, xopt, fopt) <
           wrapped_fitness_function(tempParticle2, dimensionsCount, xopt, fopt))
        {
            for(int k = 0; k < dimensionsCount; k++)
                pBests[i + k] = positions[i + k];
        }
    }
}