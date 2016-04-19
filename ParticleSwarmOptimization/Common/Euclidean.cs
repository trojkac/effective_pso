using System;
using System.Linq;

namespace Common
{
    public class Euclidean : IMetric<double[]>
    {
        public double Distance(double[] @from, double[] to)
        {
            return Math.Sqrt(@from.Select((x, i) => (x - to[i])*(x - to[i])).Sum());
        }

        public double[] VectorBetween(double[] @from, double[] to)
        {
            return @from.Select((x, i) => to[i] - x).ToArray();
        }
    }
}