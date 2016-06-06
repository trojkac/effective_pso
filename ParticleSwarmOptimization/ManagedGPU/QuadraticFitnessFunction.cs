using System;
using System.Linq;

namespace ManagedGPU
{
    internal class QuadraticFitnessFunction : ICudaFitnessFunction
    {
        public QuadraticFitnessFunction()
        {
            KernelFile = "psoKernelQuadratic.ptx";
            HostFitnessFunction = QuadraticFunction;
        }

        public Func<double[], double> HostFitnessFunction { get; private set; }
        public string KernelFile { get; private set; }

        private static double QuadraticFunction(double[] x)
        {
            return x.Sum(t => t*t);
        }
    }
}
