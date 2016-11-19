#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "bbob_generators.cuh"

__constant__ double d_OMEGA = 0.64;
__constant__ double d_phi = 1.4;

__device__ double fitness_function(double x[], int number_of_variables, double* rotation, int number_of_peaks, double* peak_values, double* x_local, double* arr_scales)
{
    size_t i, j; /* Loop over dim */
    double tmx[MAX_DIMENSIONS];
    double a = 0.1;
    double tmp2, f = 0., f_add, tmp, f_pen = 0., f_true = 0.;
    double fac;
    double result;
    double* row;
    double* row2;

    fac = -0.5 / (double)number_of_variables;

    /* Boundary handling */
    for(i = 0; i < number_of_variables; ++i)
    {
        tmp = fabs(x[i]) - 5.;
        if(tmp > 0.)
        {
            f_pen += tmp * tmp;
        }
    }
    f_add = f_pen;
    /* Transformation in search space */
    /* TODO: this should rather be done in f_gallagher */

    for(i = 0; i < number_of_variables; i++)
    {
        tmx[i] = 0;
        row = rotation + i * number_of_variables;
        for(j = 0; j < number_of_variables; ++j)
        {
            tmx[i] += row[j] * x[j];
        }
    }
    /* Computation core*/
    for(i = 0; i < number_of_peaks; ++i)
    {
        row = arr_scales + i * (int)number_of_peaks;
        tmp2 = 0.;
        for(j = 0; j < number_of_variables; ++j)
        {
            row2 = x_local + j * number_of_variables;
            tmp = (tmx[j] - row2[i]);
            tmp2 += row[j] * tmp * tmp;
        }
        tmp2 = peak_values[i] * exp(fac * tmp2);
        f = coco_double_max(f, tmp2);
    }

    f = 10. - f;
    if(f > 0)
    {
        f_true = log(f) / a;
        f_true = pow(exp(f_true + 0.49 * (sin(f_true) + sin(0.79 * f_true))), a);
    }
    else if(f < 0)
    {
        f_true = log(-f) / a;
        f_true = -pow(exp(f_true + 0.49 * (sin(0.55 * f_true) + sin(0.31 * f_true))), a);
    }
    else
        f_true = f;

    f_true *= f_true;
    f_true += f_add;
    result = f_true;
    return result;
}

__device__ double wrapped_fitness_function(double x[], int number_of_variables,
                                           double* rotation, int number_of_peaks, double* peak_values, double* x_local, double* arr_scales)
{
    double temp[1];
    temp[0] = fitness_function(x, number_of_variables, rotation, number_of_peaks, peak_values, x_local, arr_scales);

    return temp[0];
}


extern "C" {
    __global__ void generateData(int dimension,
                                 int rseed,
                                 double* rotation, 
                                 int number_of_peaks, 
                                 double* peak_values, 
                                 double* x_local, 
                                 double* arr_scales)
    {

    }

    __global__ void transposeKernel(
        double* positions,
        double* velocities,
        double* personalBests,
        double* personalBestValues,
        int particlesCount,
        int dimensionsCount,
        double* rotation, int number_of_peaks, double* peak_values, double* x_local, double* arr_scales)
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

        double newValue = wrapped_fitness_function(tempLocation, dimensionsCount, rotation, number_of_peaks, peak_values, x_local, arr_scales);

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