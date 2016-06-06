using System;
using System.Runtime.Serialization;

namespace Common
{
    [DataContract]
    public class FunctionParameters
    {
        [DataMember]
        public string FitnessFunctionType { get; set; } 
        [DataMember]
        public int Dimension { get; set; }

        [DataMember]
        public double[] Coefficients { get; set; }

        [DataMember]
        public Tuple<double, double>[] SearchSpace { get; set; }
    }
}
