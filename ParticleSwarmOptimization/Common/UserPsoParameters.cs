using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common
{
    public class UserPsoParameters
    {
        public FitnessFunctionType FitnessFunctionType { get; set; }
        public int Dimension { get; set; }
        public double[] Coefficients { get; set; }
        public Tuple<double, double>[] SearchSpace { get; set; }
    }
}
