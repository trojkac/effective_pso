#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

const int NUM_OF_DIMENSIONS = 3;

__constant__ float d_OMEGA= 0.5;
__constant__ float d_c1 = 1.5;
__constant__ float d_c2 = 1.5;
__constant__ float d_phi = 3.1415;

__device__ float tempParticle1[NUM_OF_DIMENSIONS];
__device__ float tempParticle2[NUM_OF_DIMENSIONS];

// Fungsi yang dioptimasi
// Levy 3-dimensional
__device__ float fitness_function(float x[], int dimensionsCount)
{
    float res = 0;

    for (int i = 0; i < dimensionsCount - 1; i++)
    {
        res += x[i] * x[i];
    }

    return res;
}

extern "C" {
	__global__ void kernelUpdateParticle(float *positions, float *velocities, 
										 float *pBests, float *gBest,
										 int particlesCount, int dimensionsCount,
										 float r1, float r2)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
    
		if(i >= particlesCount * dimensionsCount)
			return;

		//float rp = getRandomClamped();
		//float rg = getRandomClamped();
    
		float rp = r1;
		float rg = r2;

		velocities[i] = d_OMEGA * velocities[i] + d_c1 * rp * (pBests[i] - positions[i])
				+ d_c2 * rg * (gBest[i % dimensionsCount] - positions[i]);

		// Update posisi particle
		positions[i] += velocities[i];
	}

	__global__ void kernelUpdatePBest(float *positions, float *pBests, float* gBest,
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