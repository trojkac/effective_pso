using System.Runtime.Serialization;

namespace Common.Parameters
{
    [DataContract]
    public class FunctionParameters : IParameters
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
}
