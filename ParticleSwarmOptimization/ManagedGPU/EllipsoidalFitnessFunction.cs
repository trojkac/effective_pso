using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManagedGPU
{
    internal class EllipsoidalFitnessFunction : ICudaFitnessFunction
    {
        public EllipsoidalFitnessFunction()
        {
            KernelFile = "f2_ellipsoidal_kernel.ptx";
            HostFitnessFunction = Ellipsoidal;
        }

        public Func<double[], double> HostFitnessFunction { get; private set; }
        public string KernelFile { get; private set; }

        private static double Ellipsoidal(double[] x)
        {
            double condition = 100000;
            double result;

            result = x[0] * x[0];

            for (int i = 1; i < x.Length; i++)
            {
                double exponent = 1.0 * (double)(long)i / ((double)(long)x.Length - 1.0);
                result += Math.Pow(condition, exponent) * x[i] * x[i];
            }

            return result;
        }
    }
}
