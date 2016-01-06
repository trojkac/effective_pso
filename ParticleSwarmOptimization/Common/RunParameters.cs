using System;
using System.Collections.Generic;

namespace Common
{
    public class RunParameters
    {
        public FitnessFunctionType FitnessFunctionType { get; set; }
        public int Dimension { get; set; }
        public double[] Coefficients { get; set; }
        public int NrOfVCpu { get; set; }
        public bool IsGpu { get; set; }
        public Tuple<double, double>[] SearchSpace { get; set; }
        public List<string> PeerAddresses { get; set; }

        public RunParameters()
        {

        }
    }
}
