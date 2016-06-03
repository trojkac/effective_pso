using System;
using System.Linq;

namespace ManagedGPU
{
    internal class SimpleRastriginFitnessFunction : ICudaFitnessFunction
    {
        public SimpleRastriginFitnessFunction()
        {
            HostFitnessFunction = SimpleRastriginFunction;
            KernelFile = "psoKernelRastrigin.ptx";
        }

        public Func<double[], double> HostFitnessFunction { get; private set; }
        public string KernelFile { get; private set; }

        private static int A = 10;

        private static double SimpleRastriginFunction(double[] x)
        {
            return A*x.Length + x.Sum(t => t*t - A*Math.Cos(2*Math.PI*t));
        }
    }
}
