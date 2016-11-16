using System;
using System.Runtime.Serialization;
using System.Collections.Generic;

namespace Common.Parameters
{
    [DataContract]
    public class PsoParameters : IParameters
    {

        /// <summary>
        /// Collection specifying types of particles and number of particles of each type
        /// </summary>
        [DataMember]
        public ParticlesCount[] Particles;

        [DataMember] public int ParticlesCount;

        [DataMember] public int ParticleIterationsToRestart;

        [DataMember] public int PsoIterationsToRestart;

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

        public PsoParameters(ParticlesCount[] particlesSet, FunctionParameters functionParams)
        {
            Epsilon = 0;
            TargetValue = 0;
            Iterations = 0;
            IterationsLimitCondition = true;
            TargetValueCondition = false;
            Particles = particlesSet;
            FunctionParameters = functionParams;

        }

        public PsoParameters()
        {
            Epsilon = 0;
            TargetValue = 0;
            Iterations = 0;
            IterationsLimitCondition = true;
            TargetValueCondition = false;
            Particles = new ParticlesCount[0];
            FunctionParameters = new FunctionParameters();
        }
    }
}
