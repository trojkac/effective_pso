using System;
using System.Linq;
using ManagedCuda;

namespace ManagedGPU
{
    public static class GpuController
    {
        public static bool AnySupportedGpu()
        {
            var devicesCount = CudaContext.GetDeviceCount();

            return Enumerable
                .Range(0, devicesCount)
                .Any(deviceId => CudaContext.GetDeviceInfo(deviceId).ComputeCapability.Major >= 2);
        }

        public static Tuple<CudaParticle, GenericCudaAlgorithm> Setup(CudaParams parameters)
        {
            var proxy = CreateProxy(parameters);
            return new Tuple<CudaParticle, GenericCudaAlgorithm>(CreateParticle(proxy), CreateCudaAlgorithm(parameters, proxy));
        }

        private static StateProxy CreateProxy(CudaParams parameters)
        {
            return new StateProxy(parameters);
        }

        private static CudaParticle CreateParticle(StateProxy proxy)
        {
            var particle = new CudaParticle(proxy);
            particle.Init();
            return particle;
        }
        
        private static GenericCudaAlgorithm CreateCudaAlgorithm(CudaParams parameters, StateProxy proxy)
        {
            return CudaAlgorithmFactory.AlgorithmForFunction(parameters, proxy);
        }

        private static double Random(Random rng)
        {
            return rng.NextDouble();
        }

        private static double RandomIn(Random rng, double min, double max)
        {
            return min + Random(rng) * (max - min);
        }

        private static double HostFitnessFunction(double[] particle)
        {
            return particle.Sum(t => t*t);
        }


    }
}
