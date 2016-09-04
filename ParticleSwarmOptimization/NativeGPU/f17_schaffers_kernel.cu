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

    /* Computation core */
    result = 0.0;
    for(i = 0; i < number_of_variables - 1; ++i)
    {
        const double tmp = x[i] * x[i] + x[i + 1] * x[i + 1];
        result += pow(tmp, 0.25) * (1.0 + pow(sin(50.0 * pow(tmp, 0.1)), 2.0));
    }
    result = pow(result / ((double)(long)number_of_variables - 1.0), 2.0);

    return result;
}

__device__ double wrapped_fitness_function(double x[], int number_of_variables,
                                           double* xopt, double* M, double* b, double fopt, double asymmetric, double penalty)
{
    transform_vars_shift(x, number_of_variables, xopt);
    transform_vars_affine(x, number_of_variables, M, b);
    transform_vars_asymmetric(x, number_of_variables, asymmetric);
    transform_vars_affine(x, number_of_variables, M, b);

    double temp[1];
    temp[0] = fitness_function(x, number_of_variables);
    transform_obj_shift(temp, 1, fopt);
    transform_obj_penalize(x, 1, penalty);

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
                                 double* obj_shift_fopt,
                                 double conditioning)
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
                double exponent = 1.0 * (int)i / ((double)(long)dimension - 1.0);
                current_row[j] = rot2[i][j] * pow(sqrt(conditioning), exponent);
            }
        }
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
                                      double* xopt, double* M, double* b, double fopt, double asymmetric, double penalty)
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

        if(wrapped_fitness_function(tempParticle1, dimensionsCount, xopt, M, b, fopt, asymmetric, penalty) <
           wrapped_fitness_function(tempParticle2, dimensionsCount, xopt, M, b, fopt, asymmetric, penalty))
        {
            for(int k = 0; k < dimensionsCount; k++)
                pBests[i + k] = positions[i + k];
        }
    }
}