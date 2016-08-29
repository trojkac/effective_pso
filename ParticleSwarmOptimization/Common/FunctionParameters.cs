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
        public DimensionBound[] SearchSpace { get; set; }
    }

    public struct DimensionBound
    {
        public double Min;
        public double Max;

        public DimensionBound(double min, double max)
        {
            Min = min;
            Max = max;
        }
    }
}
