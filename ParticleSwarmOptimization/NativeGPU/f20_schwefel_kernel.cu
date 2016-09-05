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
    double penalty, sum;

    /* Boundary handling*/
    penalty = 0.0;
    for(i = 0; i < number_of_variables; ++i)
    {
        const double tmp = fabs(x[i]) - 500.0;
        if(tmp > 0.0)
            penalty += tmp * tmp;
    }

    /* Computation core */
    sum = 0.0;
    for(i = 0; i < number_of_variables; ++i)
    {
        sum += x[i] * sin(sqrt(fabs(x[i])));
    }
    result = 0.01 * (penalty + 418.9828872724339 - sum / (double)number_of_variables);

    return result;
}

__device__ double wrapped_fitness_function(double x[], int number_of_variables,
                                           double* xopt, double* M, double* b, double fopt, double* tmp1, double* tmp2, double seed)
{
    transform_vars_x_hat(x, number_of_variables, seed);
    transform_vars_scale(x, number_of_variables, 2);
    transform_vars_z_hat(x, number_of_variables, xopt);
    transform_vars_shift(x, number_of_variables, tmp2);
    transform_vars_affine(x, number_of_variables, M, b);
    transform_vars_shift(x, number_of_variables, tmp1);
    transform_vars_scale(x, number_of_variables, 100);

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
                                 double* obj_shift_fopt,
                                 double* tmp1,
                                 double* tmp2)
    {
        obj_shift_fopt[0] = bbob2009_compute_fopt(function, instance);

        double rot1[MAX_DIMENSIONS][MAX_DIMENSIONS];
        double rot2[MAX_DIMENSIONS][MAX_DIMENSIONS];

        bbob2009_compute_rotation(dimension, rot1, rseed + 1000000);
        bbob2009_compute_rotation(dimension, rot2, rseed);

        const double condition = 10.0;
        
        double* current_row;

        obj_shift_fopt[0] = bbob2009_compute_fopt(function, instance);
        bbob2009_unif(tmp1, dimension, rseed);
        for(int i = 0; i < dimension; ++i)
        {
            vars_shift_xopt[i] = 0.5 * 4.2096874637;
            if(tmp1[i] - 0.5 < 0)
            {
                vars_shift_xopt[i] *= -1;
            }
        }

        for(int i = 0; i < dimension; ++i)
        {
            vars_affine_b[i] = 0.0;
            current_row = vars_affine_m + i * dimension;
            for(int j = 0; j < dimension; ++j)
            {
                current_row[j] = 0.0;
                if(i == j)
                {
                    double exponent = 1.0 * (int)i / ((double)(long)dimension - 1);
                    current_row[j] = pow(sqrt(condition), exponent);
                }
            }
        }

        for(int i = 0; i < dimension; ++i)
        {
            tmp1[i] = -2 * fabs(vars_shift_xopt[i]);
            tmp2[i] = 2 * fabs(vars_shift_xopt[i]);
        }
    }

    __global__ void transposeKernel(
        double* positions,
        double* velocities,
        double* personalBests,
        double* personalBestValues,
        int particlesCount,
        int dimensionsCount,
        double* xopt, double* M, double* b, double fopt, double* tmp1, double* tmp2, double seed)
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

        double newValue = wrapped_fitness_function(particleLoc, dimensionsCount, xopt, M, b, fopt, tmp1, tmp2, seed);

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