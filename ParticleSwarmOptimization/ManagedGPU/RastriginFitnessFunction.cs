using System;
using System.Linq;

namespace ManagedGPU
{
    internal class RastriginFitnessFunction : ICudaFitnessFunction
    {
        public RastriginFitnessFunction()
        {
            HostFitnessFunction = RastriginFunction;
            KernelFile = "psoKernelRastrigin.ptx";
        }

        public Func<double[], double> HostFitnessFunction { get; private set; }
        public string KernelFile { get; private set; }

        private static int A = 10;

        private static double RastriginFunction(double[] x)
        {
            return A*x.Length + x.Sum(t => t*t - A*Math.Cos(2*Math.PI*t));
        }
    }
}
