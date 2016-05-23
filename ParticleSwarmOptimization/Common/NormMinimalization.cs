using System;
using System.Linq;

namespace Common
{
    public class NormMinimalization : IOptimization<double[]>
    {
        private static double Norm(double[] v)
        {
            if (v == null) throw new ArgumentNullException("v");
            return Math.Sqrt(v.Select(x => x*x).Sum());
        }

        public int IsBetter(double[] a, double[] b)
        {
            return Math.Sign(Norm(a) - Norm(b));
        }

        public double[] WorstValue(int fitnessDim)
        {
            return Enumerable.Repeat(double.PositiveInfinity, fitnessDim).ToArray();
        }

        public bool AreClose(double[] a, double[] b, double epsilon)
        {
            return Math.Abs(Norm(a) - Norm(b)) < epsilon;
        }
    }
}