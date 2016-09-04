#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "bbob_generators.cuh"

__constant__ double d_OMEGA = 0.64;
__constant__ double d_phi = 1.4;

__constant__ double PI = 3.1415;

__device__ double fitness_function(double x[], int number_of_variables, double* xopt, double fopt, double *rot1, double *rot2)
{
    const double condition = 100;
    const double alpha = 10.0;
    size_t i, j;
    double penalty = 0.0, x1;
    double result;

    double tempx[MAX_DIMENSIONS];
    double tempxx[MAX_DIMENSIONS];

    double* current_row;

    for(i = 0; i < number_of_variables; ++i)
    {
        double tmp;
        tmp = fabs(x[i]) - 5.0;
        if(tmp > 0.0)
            penalty += tmp * tmp;
    }

    for(i = 0; i < number_of_variables; ++i)
    {
        double c1;
        tempx[i] = 0.0;
        current_row = rot2 + i * number_of_variables;
        c1 = sqrt(pow(condition / 10., (double)i / (double)(number_of_variables - 1)));
        for(j = 0; j < number_of_variables; ++j)
        {
            tempx[i] += c1 * current_row[j] * (x[j] - xopt[j]);
        }
    }
    x1 = tempx[0];

    for(i = 0; i < number_of_variables; ++i)
    {
        if(fabs(tempx[i]) > 0.5)
            tempx[i] = coco_double_round(tempx[i]);
        else
            tempx[i] = coco_double_round(alpha * tempx[i]) / alpha;
    }

    for(i = 0; i < number_of_variables; ++i)
    {
        tempxx[i] = 0.0;
        current_row = rot1 + i * number_of_variables;
        for(j = 0; j < number_of_variables; ++j)
        {
            tempxx[i] += current_row[j] * tempx[j];
        }
    }

    /* Computation core */
    result = 0.0;
    for(i = 0; i < number_of_variables; ++i)
    {
        double exponent;
        exponent = (double)(long)i / ((double)(long)number_of_variables - 1.0);
        result += pow(condition, exponent) * tempxx[i] * tempxx[i];
        ;
    }
    result = 0.1 * coco_double_max(fabs(x1) * 1.0e-4, result) + penalty + fopt;

    return result;
}

__device__ double wrapped_fitness_function(double x[], int number_of_variables,
                                           double* xopt, double* rot1, double* rot2, double fopt)
{
    double temp[1];
    temp[0] = fitness_function(x, number_of_variables, xopt, fopt, rot1, rot2);
    return temp[0];
}


extern "C" {
    __global__ void generateData(int dimension,
                                 int rseed,
                                 int function,
                                 int instance,
                                 double* rot1,
                                 double* rot2,
                                 double* vars_shift_xopt,
                                 double* obj_shift_fopt)
    {
        bbob2009_compute_xopt(vars_shift_xopt, rseed, dimension);
        obj_shift_fopt[0] = bbob2009_compute_fopt(function, instance);

        double rot1d[MAX_DIMENSIONS][MAX_DIMENSIONS];
        double rot2d[MAX_DIMENSIONS][MAX_DIMENSIONS];

        bbob2009_compute_rotation(dimension, rot1d, rseed + 1000000);
        bbob2009_compute_rotation(dimension, rot2d, rseed);

        double *current_row_1;
        double *current_row_2;

        for(int i = 0; i < dimension; ++i)
        {
            current_row_1 = rot1 + i * dimension;
            current_row_2 = rot2 + i * dimension;

            for(int j = 0; j < dimension; ++j)
            {
                current_row_1[j] = rot1d[i][j];
                current_row_2[j] = rot2d[i][j];
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
                                      double* xopt, double* rot1, double* rot2, double fopt)
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

        if(wrapped_fitness_function(tempParticle1, dimensionsCount, xopt, rot1, rot2, fopt) <
           wrapped_fitness_function(tempParticle2, dimensionsCount, xopt, rot1, rot2, fopt))
        {
            for(int k = 0; k < dimensionsCount; k++)
                pBests[i + k] = positions[i + k];
        }
    }
}