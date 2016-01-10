using System;

namespace Common
{
    public class UserFunctionParameters
    {
        public FitnessFunctionType FitnessFunctionType { get; set; }
        public int Dimension { get; set; }
        public double[] Coefficients { get; set; }
        public Tuple<double, double>[] SearchSpace { get; set; }
    }
}
