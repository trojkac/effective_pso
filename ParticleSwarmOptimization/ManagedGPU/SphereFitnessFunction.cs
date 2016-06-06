using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManagedGPU
{
    internal class SphereFitnessFunction : ICudaFitnessFunction
    {
        public SphereFitnessFunction()
        {
            KernelFile = "f1_sphere_kernel.ptx";
            HostFitnessFunction = Sphere;
        }

        public Func<double[], double> HostFitnessFunction { get; private set; }
        public string KernelFile { get; private set; }

        private static double Sphere(double[] x)
        {
            return x.Sum(t => t * t);
        }
    }
}
