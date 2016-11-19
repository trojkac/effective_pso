#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "bbob_generators.cuh"

__constant__ double d_OMEGA = 0.64;
__constant__ double d_phi = 1.4;

__device__ double fitness_function(double x[], int number_of_variables, double* xopt, double* rot1, double* rot2)
{
    double result;
    const double condition = 100.;
    size_t i, j;
    double penalty = 0.0;
    const double mu0 = 2.5;
    const double d = 1.;
    const double s = 1. - 0.5 / (sqrt((double)(number_of_variables + 20)) - 4.1);
    const double mu1 = -sqrt((mu0 * mu0 - d) / s);
    double sum1 = 0., sum2 = 0., sum3 = 0.;
    double tmpvect[MAX_DIMENSIONS];

    double x_hat[MAX_DIMENSIONS];
    double z[MAX_DIMENSIONS];

    for(i = 0; i < number_of_variables; ++i)
    {
        double tmp;
        tmp = fabs(x[i]) - 5.0;
        if(tmp > 0.0)
            penalty += tmp * tmp;
    }

    /* x_hat */
    for(i = 0; i < number_of_variables; ++i)
    {
        x_hat[i] = 2. * x[i];
        if(xopt[i] < 0.)
        {
            x_hat[i] *= -1.;
        }
    }

    double *row;

    
    /* affine transformation */
    for(i = 0; i < number_of_variables; ++i)
    {
        double c1;
        tmpvect[i] = 0.0;
        row = rot2 + i * number_of_variables;
        c1 = pow(sqrt(condition), ((double)i) / (double)(number_of_variables - 1));
        for(j = 0; j < number_of_variables; ++j)
        {
            tmpvect[i] += c1 * row[j] * (x_hat[j] - mu0);
        }
    }
    for(i = 0; i < number_of_variables; ++i)
    {
        z[i] = 0;
        row = rot1 + i * number_of_variables;
        for(j = 0; j < number_of_variables; ++j)
        {
            z[i] += row[j] * tmpvect[j];
        }
    }
    /* Computation core */
    for(i = 0; i < number_of_variables; ++i)
    {
        sum1 += (x_hat[i] - mu0) * (x_hat[i] - mu0);
        sum2 += (x_hat[i] - mu1) * (x_hat[i] - mu1);
        sum3 += cos(coco_two_pi * z[i]);
    }
    result = coco_double_min(sum1, d * (double)number_of_variables + s * sum2)
        + 10. * ((double)number_of_variables - sum3) + 1e4 * penalty;

    return result;
}

__device__ double wrapped_fitness_function(double x[], int number_of_variables,
                                           double* xopt, double* rot1, double* rot2)
{
    double temp[1];
    temp[0] = fitness_function(x, number_of_variables, xopt, rot1, rot2);
    return temp[0];
}


extern "C" {
    __global__ void generateData(int dimension,
                                 int rseed,
                                 double* rot1,
                                 double* rot2,
                                 double* vars_shift_xopt)
    {
        bbob2009_compute_xopt(vars_shift_xopt, rseed, dimension);

        double rot1d[MAX_DIMENSIONS][MAX_DIMENSIONS];
        double rot2d[MAX_DIMENSIONS][MAX_DIMENSIONS];

        bbob2009_compute_rotation(dimension, rot1d, rseed + 1000000);
        bbob2009_compute_rotation(dimension, rot2d, rseed);

        for(int i = 0; i < dimension; i++)
        {
            double* row1 = rot1 + i * dimension;
            double* row2 = rot2 + i * dimension;

            for(int j = 0; j < dimension; j++)
            {
                row1[j] = rot1d[i][j];
                row2[j] = rot2d[i][j];
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
        double* xopt, double* rot1, double* rot2)
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

        double newValue = wrapped_fitness_function(tempLocation, dimensionsCount, xopt, rot1, rot2);

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