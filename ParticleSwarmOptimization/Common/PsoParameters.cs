using System;
using System.Runtime.Serialization;

namespace Common
{
    [DataContract]
    public struct PsoParameters
    {

        /// <summary>
        /// Collection specifying types of particles and number of particles of each type
        /// </summary>
        [DataMember]
        public Tuple<PsoParticleType, int>[] Particles;

        [DataMember(IsRequired = true)]
        public bool IterationsLimitCondition;

        /// <summary>
        /// Iterations limit.
        /// </summary>
        [DataMember]
        public int Iterations;

        [DataMember(IsRequired = true)]
        public bool TargetValueCondition;
        
        /// <summary>
        /// If result of the optimization problem is known it can be used as a stop condition
        /// </summary>
        [DataMember]
        public double TargetValue;

        /// <summary>
        /// Specifies how accurate should the result be if the target value is given
        /// </summary>
        [DataMember]
        public double Epsilon;

        [DataMember]
        public FunctionParameters FunctionParameters;

        public PsoParameters(Tuple<PsoParticleType, int>[] particlesSet, FunctionParameters functionParams)
        {
            Epsilon = 0;
            TargetValue = 0;
            Iterations = 0;
            IterationsLimitCondition = true;
            TargetValueCondition = false;
            Particles = new Tuple<PsoParticleType, int>[2];
            FunctionParameters = functionParams;
            Particles = particlesSet;
        }
    }
}
