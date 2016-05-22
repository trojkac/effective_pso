using System;
using System.Linq;
using ManagedCuda;

namespace ManagedGPU
{
    public static class GpuController
    {
        public static Tuple<CudaParticle, CudaAlgorithm> Setup(CudaParams parameters)
        {
            var proxy = CreateProxy(parameters);
            return new Tuple<CudaParticle, CudaAlgorithm>(CreateParticle(proxy), CreateCudaAlgorithm(parameters, proxy));
        }

        private static StateProxy CreateProxy(CudaParams parameters)
        {
            return new StateProxy(parameters);
        }

        private static CudaParticle CreateParticle(StateProxy proxy)
        {
            return new CudaParticle(proxy);
        }
        
        private static CudaAlgorithm CreateCudaAlgorithm(CudaParams parameters, StateProxy proxy)
        {
            return new CudaAlgorithm(parameters, proxy);
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
