using System;

namespace Common
{
    public class FirstValueOptimization : IOptimization<double[]>
    {
        public int IsBetter(double[] a, double[] b)
        {
            return Math.Sign(a[0] - b[0]);
        }

        public double[] WorstValue(int fitnessDim)
        {
            return new[] { double.PositiveInfinity };
        }

        public bool AreClose(double[] a, double[] b, double epsilon)
        {
            return Math.Abs(a[0] - b[0]) < epsilon;
        }
    }
}