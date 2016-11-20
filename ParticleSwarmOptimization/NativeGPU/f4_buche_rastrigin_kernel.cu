#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "bbob_generators.cuh"




__device__ double fitness_function(double x[], int number_of_variables)
{
    double tmp = 0., tmp2 = 0.;
    int i;
    double result;

    result = 0.0;
    for(i = 0; i < number_of_variables; ++i)
    {
        tmp += cos(coco_two_pi * x[i]);
        tmp2 += x[i] * x[i];
    }
    result = 10.0 * ((double)(long)number_of_variables - tmp) + tmp2 + 0;
    return result;
}

__device__ double wrapped_fitness_function(double x[], int number_of_variables,
                                           double* xopt, double fopt, double penalty)
{
    transform_vars_shift(x, number_of_variables, xopt);
    transform_vars_oscillate(x, number_of_variables);
    transform_vars_brs(x, number_of_variables);
    double temp[1];
    temp[0] = fitness_function(x, number_of_variables);
    transform_obj_shift(temp, 1, fopt);
    transform_obj_penalize(temp, 1, penalty);

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

        for(int i = 0; i < dimension; i += 2)
            vars_shift_xopt[i] = fabs(vars_shift_xopt[i]);
        
        obj_shift_fopt[0] = bbob2009_compute_fopt(function, instance);
    }

    __global__ void transposeKernel(
        double* positions,
        double* velocities,
        double* personalBests,
        double* personalBestValues,
        int particlesCount,
        int dimensionsCount,
        double* xopt, double fopt, double penalty)
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

        double newValue = wrapped_fitness_function(tempLocation, dimensionsCount, xopt, fopt, penalty);

        if(newValue < personalBestValues[i])
        {
            personalBestValues[i] = newValue;

            double* particlePersonalBest = personalBests + i * dimensionsCount;

            for(int i = 0; i < dimensionsCount; i++)
                particlePersonalBest[i] = particleLoc[i];
        }
    }
}