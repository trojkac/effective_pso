using System.Runtime.Serialization;

namespace Common
{
    [DataContract]
    public class ParticleState
    {
        [DataMember]
        public double FitnessValue { get; set; }
        [DataMember]
        public double[] Location { get; set; }

        public ParticleState(double[] location, double fitnessValue)
        {
            FitnessValue = fitnessValue;
            Location = location;
        }
    }
}