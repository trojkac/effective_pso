#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "bbob_generators.cuh"




__device__ double fitness_function(double x[], int number_of_variables)
{
    size_t i, j;
    double tmp, tmp2;
    double result;

    /* Computation core */
    result = 1.0;
    for(i = 0; i < number_of_variables; ++i)
    {
        tmp = 0;
        for(j = 1; j < 33; ++j)
        {
            tmp2 = pow(2., (double)j);
            tmp += fabs(tmp2 * x[i] - coco_double_round(tmp2 * x[i])) / tmp2;
        }
        tmp = 1.0 + ((double)(long)i + 1) * tmp;
        result *= tmp;
    }
    result = 10. / ((double)number_of_variables) / ((double)number_of_variables)
        * (-1. + pow(result, 10. / pow((double)number_of_variables, 1.2)));

    return result;
}

__device__ double wrapped_fitness_function(double x[], int number_of_variables,
                                           double* xopt, double* M, double* b, double fopt, double penalty)
{
    transform_vars_shift(x, number_of_variables, xopt);
    transform_vars_affine(x, number_of_variables, M, b);
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
                    current_row[j] += rot1[i][k] * pow(sqrt(100.0), exponent) * rot2[k][j];
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
        double* xopt, double* M, double* b, double fopt, double penalty)
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

        double newValue = wrapped_fitness_function(tempLocation, dimensionsCount, xopt, M, b, fopt, penalty);

        if(newValue < personalBestValues[i])
        {
            personalBestValues[i] = newValue;

            double* particlePersonalBest = personalBests + i * dimensionsCount;

            for(int i = 0; i < dimensionsCount; i++)
                particlePersonalBest[i] = particleLoc[i];
        }
    }
}