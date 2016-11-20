#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "bbob_generators.cuh"

extern "C" {

	__global__ void updateVelocityKernel(
		double* positions,
		double* velocities,
		double* personalBests,
		double* personalBestValues,
		int* neighbors,
		int particlesCount,
		int dimensionsCount,
		double* phis1,
		double* phis2
		)
	{
		int particleId = blockIdx.x * blockDim.x + threadIdx.x;

		if (particleId >= particlesCount) return;

		double* particleLoc = positions + particleId * dimensionsCount;
		double* particleVel = velocities + particleId * dimensionsCount;
		double* particleBest = personalBests + particleId * dimensionsCount;
		double particleBestValue = personalBestValues[particleId];

		int* particleNeighbors = neighbors + particleId * 2;

		int leftNeighborId = particleNeighbors[0];
		double* leftNeighborBest = personalBests + leftNeighborId * dimensionsCount;
		double leftNeighborBestVal = personalBestValues[leftNeighborId];

		int rightNeighborId = particleNeighbors[1];
		double* rightNeighborBest = personalBests + rightNeighborId * dimensionsCount;
		double rightNeighborBestVal = personalBestValues[rightNeighborId];

		double* globalBest = particleBest;
		double globalBestVal = particleBestValue;

		if (leftNeighborBestVal < globalBestVal)
		{
			globalBest = leftNeighborBest;
			globalBestVal = leftNeighborBestVal;
		}

		if (rightNeighborBestVal < globalBestVal)
		{
			globalBest = rightNeighborBest;
			globalBestVal = rightNeighborBestVal;

		}

		double toPersonalBest[MAX_DIMENSIONS];
		vector_between(particleLoc, particleBest, dimensionsCount, toPersonalBest);

		double toGlobalBest[MAX_DIMENSIONS];
		vector_between(particleLoc, globalBest, dimensionsCount, toGlobalBest);
		double* phi1 = phis1 + dimensionsCount*particleId;
		double* phi2 = phis2 + dimensionsCount*particleId;
		for (int i = 0; i < dimensionsCount; i++)
		{
			particleVel[i] = particleVel[i] * d_OMEGA + d_phi * phi1[i] * toGlobalBest[i] + d_phi * phi2[i] * toPersonalBest[i];
		}
	}
}