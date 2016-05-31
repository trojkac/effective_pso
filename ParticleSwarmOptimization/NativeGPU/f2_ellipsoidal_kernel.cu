#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

const int NUM_OF_DIMENSIONS = 3;

__constant__ double d_OMEGA= 0.64;
__constant__ double d_phi = 1.4;

__device__ double tempParticle1[NUM_OF_DIMENSIONS];
__device__ double tempParticle2[NUM_OF_DIMENSIONS];

// Simple quadratic function
__device__ double fitness_function(double x[], int dimensionsCount)
{
	const double condition = 1.0e6;
	size_t i = 0;
	double result;

	result = x[i] * x[i];
	for (i = 1; i < dimensionsCount; ++i) {
		const double exponent = 1.0 * (double) (long) i / ((double) (long) dimensionsCount - 1.0);
		result += pow((double)condition, (double)exponent) * x[i] * x[i];
	}

	return result;
}

extern "C" {
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
									  int particlesCount, int dimensionsCount)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
    
		if(i >= particlesCount * dimensionsCount || i % dimensionsCount != 0)
			return;

		for (int j = 0; j < dimensionsCount; j++)
		{
			tempParticle1[j] = positions[i + j];
			tempParticle2[j] = pBests[i + j];
		}

		if (fitness_function(tempParticle1, dimensionsCount) < fitness_function(tempParticle2, dimensionsCount))
		{
			for (int k = 0; k < dimensionsCount; k++)
				pBests[i + k] = positions[i + k];
		}
	}
}