#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "bbob_generators.cuh"




__device__ double fitness_function(double x[], int number_of_variables)
{
    size_t i = 0;
    double tmp = 0;
    double result;

    /* Computation core */
    result = 0.0;
    for(i = 0; i < number_of_variables - 1; ++i)
    {
        const double c1 = x[i] * x[i] - x[i + 1];
        const double c2 = 1.0 - x[i];
        tmp = 100.0 * c1 * c1 + c2 * c2;
        result += tmp / 4000. - cos(tmp);
    }
    result = 10. + 10. * result / (double)(number_of_variables - 1);

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
        for(int i = 0; i < dimension; ++i)
        {
            vars_shift_xopt[i] = -0.5;
        }
        obj_shift_fopt[0] = bbob2009_compute_fopt(function, instance);

        double rot1[MAX_DIMENSIONS][MAX_DIMENSIONS];

        bbob2009_compute_rotation(dimension, rot1, rseed);

        double scales = coco_double_max(1.0, sqrt((double)dimension) / 8.0);
        double *current_row;

        for(int i = 0; i < dimension; ++i)
        {
            for(int j = 0; j < dimension; ++j)
            {
                rot1[i][j] *= scales;
            }
        }

        bbob2009_copy_rotation_matrix(rot1, vars_affine_m, vars_affine_b, dimension);
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
}