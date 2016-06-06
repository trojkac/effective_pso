using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ManagedGPU
{
    internal class BucheRastriginFitnessFunction : ICudaFitnessFunction
    {
        public BucheRastriginFitnessFunction()
        {
            KernelFile = "f4_buche_rastrigin_kernel.ptx";
            HostFitnessFunction = BucheRastrigin;
        }

        public Func<double[], double> HostFitnessFunction { get; private set; }
        public string KernelFile { get; private set; }

        private static double BucheRastrigin(double[] x)
        {
            double tmp = 0.0, tmp2 = 0.0;
            double result = 0.0;

            for (int i = 0; i < x.Length; ++i)
            {
                tmp += Math.Cos(2 * Math.PI * x[i]);
                tmp2 += x[i] * x[i];
            }

            result = 10.0 * ((double)(long)x.Length - tmp) + tmp2;

            return result;
        }
    }
}
