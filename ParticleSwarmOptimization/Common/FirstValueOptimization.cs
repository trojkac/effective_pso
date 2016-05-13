using System;

namespace Common
{
    public class FirstValueOptimization : IOptimization<double[]>
    {
        public int IsBetter(double[] a, double[] b)
        {
            return Math.Sign(a[0] - b[0]);
        }
    }
}