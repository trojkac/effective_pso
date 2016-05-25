using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;

namespace Common
{
    [DataContract]
    public struct ParticleState : IState<double[],double[]>
    {
        [DataMember]
        public double[] FitnessValue
        {
            get { return _fitnessValue; }
            set { _fitnessValue = value; }
        }
        [DataMember]
        public double[] Location
        {
            get { return _location; }
            set { _location = value; }
        }


        private double[] _location;
        private double[] _fitnessValue;

        public ParticleState(double[] location, double[] fitness ) : this()
        {
            FitnessValue = fitness;
            Location = location;
        }
    }
}