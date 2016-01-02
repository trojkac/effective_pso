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

        public static ParticleState WorstState
        {
            get
            {
                return new ParticleState(null, double.NegativeInfinity);
            }
        }

        public ParticleState(double[] location, double fitnessValue)
        {
            FitnessValue = fitnessValue;
            Location = location ;
        }

        public ParticleState()
        {
            var worst = WorstState;
            FitnessValue = worst.FitnessValue;
            Location = worst.Location;
        }
    }
}