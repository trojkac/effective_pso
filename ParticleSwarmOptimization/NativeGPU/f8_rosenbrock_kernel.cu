#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "bbob_generators.cuh"




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
                                           double* xopt, double fopt, double *minus_ones, double factor)
{
    transform_vars_shift(x, number_of_variables, xopt);
    transform_vars_scale(x, number_of_variables, factor);
    transform_vars_shift(x, number_of_variables, minus_ones);
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
        
        for(int i = 0; i < dimension; ++i)
        {
            vars_shift_xopt[i] *= 0.75;
        }

        obj_shift_fopt[0] = bbob2009_compute_fopt(function, instance);
    }

    __global__ void transposeKernel(
        double* positions,
        double* velocities,
        double* personalBests,
        double* personalBestValues,
        int particlesCount,
        int dimensionsCount,
        double* xopt, double fopt, double* minus_ones, double factor)
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

        double newValue = wrapped_fitness_function(tempLocation, dimensionsCount, xopt, fopt, minus_ones, factor);

        if(newValue < personalBestValues[i])
        {
            personalBestValues[i] = newValue;

            double* particlePersonalBest = personalBests + i * dimensionsCount;

            for(int i = 0; i < dimensionsCount; i++)
                particlePersonalBest[i] = particleLoc[i];
        }
    }
}