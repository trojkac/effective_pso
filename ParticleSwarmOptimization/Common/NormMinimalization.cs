using System;
using System.Linq;

namespace Common
{
    public class NormMinimalization : IOptimization<double[]>
    {
        private static double Norm(double[] v)
        {
            return Math.Sqrt(v.Select(x => x*x).Sum());
        }
        public int IsBetter(double[] a, double[] b)
        {
            return Math.Sign(Norm(a) - Norm(b));
        }
    }
}