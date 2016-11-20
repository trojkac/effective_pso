#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "bbob_generators.cuh"

#define F_WEIERSTRASS_SUMMANDS 12




__device__ double fitness_function(double x[], int number_of_variables, double* ak, double* bk, double f0)
{
    size_t i, j;
    double result;

    result = 0.0;
    for(i = 0; i < number_of_variables; ++i)
    {
        for(j = 0; j < F_WEIERSTRASS_SUMMANDS; ++j)
        {
            result += cos(coco_two_pi * (x[i] + 0.5) * bk[j]) * ak[j];
        }
    }
    result = 10.0 * pow(result / (double)(long)number_of_variables - f0, 3.0);

    return result;
}

__device__ double wrapped_fitness_function(double x[], int number_of_variables,
                                           double* xopt, double fopt, double penalty, double* M, double* b, double* ak, double* bk, double f0)
{
    transform_vars_shift(x, number_of_variables, xopt);
    transform_vars_affine(x, number_of_variables, M, b);
    transform_obj_oscillate(x, number_of_variables);
    transform_vars_affine(x, number_of_variables, M, b);
    double temp[1];
    temp[0] = fitness_function(x, number_of_variables, ak, bk, f0);
    transform_obj_shift(temp, 1, fopt);
    transform_obj_penalize(x, 1, penalty);

    return temp[0];
}


extern "C" {
    __global__ void generateData(int dimension,
                                 int rseed,
                                 int function,
                                 int instance,
                                 double* M,
                                 double* b,
                                 double* vars_shift_xopt,
                                 double* obj_shift_fopt)
    {
        bbob2009_compute_xopt(vars_shift_xopt, rseed, dimension);
        obj_shift_fopt[0] = bbob2009_compute_fopt(function, instance);


        double rot1[MAX_DIMENSIONS][MAX_DIMENSIONS];
        double rot2[MAX_DIMENSIONS][MAX_DIMENSIONS];

        bbob2009_compute_rotation(dimension, rot1, rseed + 1000000);
        bbob2009_compute_rotation(dimension, rot2, rseed);

        const double condition = 100.0;
        double *current_row;

        for(int i = 0; i < dimension; ++i)
        {
            b[i] = 0.0;
            current_row = M + i * dimension;
            for(int j = 0; j < dimension; ++j)
            {
                current_row[j] = 0.0;
                for(int k = 0; k < dimension; ++k)
                {
                    const double base = 1.0 / sqrt(condition);
                    const double exponent = 1.0 * (int)k / ((double)(long)dimension - 1.0);
                    current_row[j] += rot1[i][k] * pow(base, exponent) * rot2[k][j];
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
        double* xopt, double fopt, double penalty, double* M, double* b, double* ak, double* bk, double f0)
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

        double newValue = wrapped_fitness_function(tempLocation, dimensionsCount, xopt, fopt, penalty, M, b, ak, bk, f0);

        if(newValue < personalBestValues[i])
        {
            personalBestValues[i] = newValue;

            double* particlePersonalBest = personalBests + i * dimensionsCount;

            for(int i = 0; i < dimensionsCount; i++)
                particlePersonalBest[i] = particleLoc[i];
        }
    }
}