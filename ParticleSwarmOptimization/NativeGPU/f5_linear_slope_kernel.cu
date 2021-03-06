#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "bbob_generators.cuh"




__device__ double fitness_function(double x[], int number_of_variables, double* xopt)
{
    const double alpha = 100.0;
    size_t i;
    double result = 0.0;

    for(i = 0; i < number_of_variables; ++i)
    {
        double base, exponent, si;

        base = sqrt(alpha);
        exponent = (double)(long)i / ((double)(long)number_of_variables - 1);
        if(xopt[i] > 0.0)
        {
            si = pow(base, exponent);
        }
        else
        {
            si = -pow(base, exponent);
        }
        result += 5.0 * fabs(si) - si * x[i];
    }

    return result;
}

__device__ double wrapped_fitness_function(double x[], int number_of_variables,
                                           double* xopt, double fopt)
{
    transform_vars_shift(x, number_of_variables, xopt);
    double temp[1];
    temp[0] = fitness_function(x, number_of_variables, xopt);
    transform_obj_shift(temp, 1, fopt);
    transform_obj_power(temp, 1);
    transform_obj_oscillate(temp, 1);

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
            if(vars_shift_xopt[i] < 0.0)
            {
                vars_shift_xopt[i] = -5.0;
            }
            else
            {
                vars_shift_xopt[i] = 5.0;
            }
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
        double* xopt, double fopt)
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

        double newValue = wrapped_fitness_function(tempLocation, dimensionsCount, xopt, fopt);

        if(newValue < personalBestValues[i])
        {
            personalBestValues[i] = newValue;

            double* particlePersonalBest = personalBests + i * dimensionsCount;

            for(int i = 0; i < dimensionsCount; i++)
                particlePersonalBest[i] = particleLoc[i];
        }
    }
}