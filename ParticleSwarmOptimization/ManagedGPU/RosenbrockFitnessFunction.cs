using System;

namespace ManagedGPU
{
    internal class RosenbrockFitnessFunction : ICudaFitnessFunction
    {
        public RosenbrockFitnessFunction()
        {
            HostFitnessFunction = RosenbrockFunction;
            KernelFile = "psoKernelRosenbrock.ptx";
        }

        public Func<double[], double> HostFitnessFunction { get; private set; }
        public string KernelFile { get; private set; }

        private static double RosenbrockFunction(double[] x)
        {
            var result = 0.0d;

            for (var i = 0; i < x.Length - 1; i++)
            {
                result += (1 - x[i])*(1 - x[i]) + 100*(x[i + 1] - x[i]*x[i])*(x[i + 1] - x[i]*x[i]);
            }

            return result;
        }
    }
}
