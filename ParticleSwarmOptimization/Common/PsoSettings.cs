using System;
using System.Runtime.Serialization;

namespace Common
{
    /// <summary>
    /// Computes fitness function
    /// </summary>
    /// <param name="values">fitness function argument which is N dimensional vector</param>
    /// <returns></returns>
    public delegate double[] FitnessFunctionEvaluation(double[] values);
    [DataContract]
    public struct PsoSettings
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

        /// <summary>
        /// Number of dimensions
        /// </summary>
        [DataMember]
        public int Dimensions;

        [DataMember]
        public UserFunctionParameters FunctionParameters;

        public PsoSettings(UserPsoParameters psoParams, UserFunctionParameters functionParams)
        {
            Epsilon = 0;
            TargetValue = 0;
            Iterations = 0;
            Particles = new Tuple<PsoParticleType, int>[2];
            FunctionParameters = functionParams;
            Particles[0] = new Tuple<PsoParticleType, int>(PsoParticleType.Standard, psoParams.StandardParticles);
            Particles[1] = new Tuple<PsoParticleType, int>(PsoParticleType.FullyInformed, psoParams.FullyInformedParticles);


            IterationsLimitCondition = psoParams.IterationsLimitCondition;
            if (IterationsLimitCondition)
            {
                Iterations = psoParams.Iterations;
            }

            TargetValueCondition = psoParams.TargetValueCondition;
            if (TargetValueCondition)
            {
                TargetValue = psoParams.TargetValue;
                Epsilon = psoParams.Epsilon;
            }

            Dimensions = functionParams.Dimension;

        }
    }
}
