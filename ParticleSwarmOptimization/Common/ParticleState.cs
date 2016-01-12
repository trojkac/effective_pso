using System.Collections.Generic;
using System.Linq;
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

       

        public static ParticleState WorstState(int dim)
        {
            return new ParticleState(Enumerable.Repeat(0.0,dim).ToArray(), double.NegativeInfinity);

        }

        public ParticleState(double[] location, double fitnessValue)
        {
            FitnessValue = fitnessValue;
            Location = location ;
        }

        public ParticleState(int dim)
        {
            var worst = WorstState(dim);
            FitnessValue = worst.FitnessValue;
            Location = worst.Location;
        }

        public ParticleState()
        {
            // TODO: Complete member initialization
        }
    }
}