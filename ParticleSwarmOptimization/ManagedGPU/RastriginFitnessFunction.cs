using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManagedGPU
{
    internal class RastriginFitnessFunction : ICudaFitnessFunction
    {
        public RastriginFitnessFunction()
        {
            KernelFile = "f3_rastrigin_kernel.ptx";
            HostFitnessFunction = Rastrigin;
        }

        public Func<double[], double> HostFitnessFunction { get; private set; }
        public string KernelFile { get; private set; }

        private static double Rastrigin(double[] x)
        {
            const int A = 10;

            return A * x.Length + x.Sum(t => t * t - A * Math.Cos(2 * Math.PI * t));
        }
    }
}
