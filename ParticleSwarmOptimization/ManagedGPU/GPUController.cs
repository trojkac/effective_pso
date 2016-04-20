using System;
using ManagedCuda;

namespace ManagedGPU
{
    public static class GpuController
    {
        public static CudaParticle Setup()
        {

            return null;
        }

        private static void CreateCudaAlgorithm()
        {
            
        }

        public static void RunCudaPSO()
        {
            const int particlesCount = 512;
            const int dimensionsCount = 3;
            const int maxIterations = 30000;
            const int size = particlesCount * dimensionsCount;

            const int threadsNum = 32;
            const int blocksNum = particlesCount / threadsNum;

            var rng = new Random();
            var ctx = new CudaContext(0);

            var updateParticle = ctx.LoadKernel("psoKernel.ptx", "kernelUpdateParticle");
            updateParticle.GridDimensions = blocksNum;
            updateParticle.BlockDimensions = threadsNum;

            var updatePersonalBest = ctx.LoadKernel("psoKernel.ptx", "kernelUpdatePBest");
            updatePersonalBest.GridDimensions = blocksNum;
            updatePersonalBest.BlockDimensions = threadsNum;

            var temp = new float[dimensionsCount];

            var hostPositions = new float[size];
            var hostVelocities = new float[size];
            var hostPersonalBests = new float[size];
            var hostGlobalBests = new float[dimensionsCount];

            for (var i = 0; i < size; i++)
            {
                hostPositions[i] = RandomIn(rng, 3.0f, 5.0f);
                hostPersonalBests[i] = hostPositions[i];
                hostVelocities[i] = 0.0f;
            }

            for (var i = 0; i < dimensionsCount; i++)
                hostGlobalBests[i] = hostPersonalBests[i];

            CudaDeviceVariable<float> devicePositions = hostPositions;
            CudaDeviceVariable<float> deviceVelocities = hostVelocities;
            CudaDeviceVariable<float> devicePersonalBests = hostPersonalBests;
            CudaDeviceVariable<float> deviceGlobalBests = hostGlobalBests;

            for (var iter = 0; iter < maxIterations; iter++)
            {
                updateParticle.Run(
                    devicePositions.DevicePointer,
                    deviceVelocities.DevicePointer,
                    devicePersonalBests.DevicePointer,
                    deviceGlobalBests.DevicePointer,
                    particlesCount,
                    dimensionsCount,
                    Random(rng),
                    Random(rng)
                );

                updatePersonalBest.Run(
                    devicePositions.DevicePointer,
                    devicePersonalBests.DevicePointer,
                    deviceGlobalBests.DevicePointer,
                    particlesCount,
                    dimensionsCount
                );

                hostPersonalBests = devicePersonalBests;

                for (var i = 0; i < size; i += dimensionsCount)
                {
                    for (var k = 0; k < dimensionsCount; k++)
                        temp[k] = hostPersonalBests[i + k];

                    if (HostFitnessFunction(temp, dimensionsCount) < HostFitnessFunction(hostGlobalBests, dimensionsCount))
                    {
                        for (var k = 0; k < dimensionsCount; k++)
                            hostGlobalBests[k] = temp[k];
                    }
                }

                deviceGlobalBests = hostGlobalBests;
            }

            hostGlobalBests = deviceGlobalBests;

            for (var i = 0; i < dimensionsCount; i++)
                Console.WriteLine("x{0} = {1}", i, hostGlobalBests[i]);

            Console.WriteLine("Minimum: {0}", HostFitnessFunction(hostGlobalBests, dimensionsCount));
        }

        private static float Random(Random rng)
        {
            return (float)rng.NextDouble();
        }

        private static float RandomIn(Random rng, float min, float max)
        {
            return min + Random(rng) * (max - min);
        }

        private static float HostFitnessFunction(float[] x, int dimensionCount)
        {
            var res = 0.0f;

            for (var i = 0; i < dimensionCount; i++)
                res += x[i] * x[i];

            return res;
        }
    }
}
