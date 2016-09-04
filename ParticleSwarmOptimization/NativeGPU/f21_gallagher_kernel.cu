#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "bbob_generators.cuh"

__constant__ double d_OMEGA = 0.64;
__constant__ double d_phi = 1.4;

__constant__ double PI = 3.1415;

__device__ double fitness_function(double x[], int number_of_variables, double* rotation, double number_of_peaks, double* peak_values, double* x_local, double* arr_scales)
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
                                           double* rotation, double number_of_peaks, double* peak_values, double* x_local, double* arr_scales)
{
    double temp[1];
    temp[0] = fitness_function(x, number_of_variables, rotation, number_of_peaks, peak_values, x_local, arr_scales);

    return temp[0];
}


extern "C" {
    __global__ void generateData(int dimension,
                                 int rseed,
                                 double* rotation, 
                                 double number_of_peaks, 
                                 double* peak_values, 
                                 double* x_local, 
                                 double* arr_scales)
    {
        size_t i, j, k;
        double maxcondition = 1000.0;
        double maxcondition1 = 1000.0;
        double b, c;
        double random_numbers[101 * MAX_DIMENSIONS];
        double fitvalues[2] = { 1.1, 9.1 };

        if(number_of_peaks == 101.0)
        {
            maxcondition1 = sqrt(maxcondition1);
            b = 10.;
            c = 5.;
        }
        else if(number_of_peaks == 21.0)
        {
            b = 9.8;
            c = 4.9;
        }

        double rot[MAX_DIMENSIONS][MAX_DIMENSIONS];

        bbob2009_compute_rotation(dimension, rot, rseed);

        double* row;

        for(i = 0; i < dimension; i++)
        {
            row = rotation + i * dimension;
            for(j = 0; j < dimension; i++)
            {
                row[j] = rot[i][j];
            }
        }

        double arrCondition[101];
        arrCondition[0] = maxcondition1;
        peak_values[0] = 10;

        for(i = 1; i < number_of_peaks; ++i)
        {
            arrCondition[i] = pow(maxcondition, (double)(i) / ((double)(number_of_peaks - 2)));
            peak_values[i] = (double)(i - 1) / (double)(number_of_peaks - 2) * (fitvalues[1] - fitvalues[0])
                + fitvalues[0];
        }

        for(i = 0; i < number_of_peaks; ++i)
        {
            row = arr_scales + i * (int)number_of_peaks;
            for(j = 0; j < dimension; ++j)
            {
                row[j] = pow(arrCondition[i],
                                             (j / ((double)(dimension - 1)) - 0.5));
            }
        }

        bbob2009_unif(random_numbers, dimension * number_of_peaks, rseed);
        for(i = 0; i < dimension; ++i)
        {
            row = x_local + i * dimension;
            double* rotrow = rotation + i * dimension;
            for(j = 0; j < number_of_peaks; ++j)
            {
                row[j] = 0.;
                for(k = 0; k < dimension; ++k)
                {
                    row[j] += rotrow[k] * (b * random_numbers[j * dimension + k] - c);
                }
                if(j == 0)
                {
                    row[j] *= 0.8;
                }
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
                                      double* rotation, double number_of_peaks, double* peak_values, double* x_local, double* arr_scales)
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

        if(wrapped_fitness_function(tempParticle1, dimensionsCount, rotation, number_of_peaks, peak_values, x_local, arr_scales) <
           wrapped_fitness_function(tempParticle2, dimensionsCount, rotation, number_of_peaks, peak_values, x_local, arr_scales))
        {
            for(int k = 0; k < dimensionsCount; k++)
                pBests[i + k] = positions[i + k];
        }
    }
}