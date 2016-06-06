using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManagedGPU
{
    internal class LinearSlopeFitnessFunction : ICudaFitnessFunction
    {
        public LinearSlopeFitnessFunction()
        {
            KernelFile = "f5_linear_slope_kernel.ptx";
            HostFitnessFunction = LinearSlope;
        }

        public Func<double[], double> HostFitnessFunction { get; private set; }
        public string KernelFile { get; private set; }

        private static double LinearSlope(double[] x)
        {
            const double alpha = 100.0;
            double result = 0.0;

            for (int i = 0; i < x.Length; i++)
            {
                double bse, exponent, si;

                bse = Math.Sqrt(alpha);
                exponent = (double)(long)i / ((double)(long)x.Length - 1);

                si = -Math.Pow(bse, exponent);

                result += 5.0 * Math.Abs(si) - si * x[i];
            }

            return result;
        }
    }
}
