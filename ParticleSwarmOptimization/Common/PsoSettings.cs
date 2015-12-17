using System;
using System.Runtime.Serialization;

namespace Common
{
    /// <summary>
    /// Computes fitness function
    /// </summary>
    /// <param name="values">fitness function argument which is N dimensional vector</param>
    /// <returns></returns>
    public delegate double FitnessFunction(double[] values); 
    [DataContract]
    public struct PsoSettings
    {

        [DataMember] 
        public FitnessFunction FitnessFunction;

        /// <summary>
        /// Collection specifying types of particles and number of particles of each type
        /// </summary>
        [DataMember]
        public Tuple<PsoParticleType, int>[] Particles;
        
        /// <summary>
        /// Nx2 array where N is dimension of the search space.
        /// In this array minimum and maximum of each dimension are stored.
        /// </summary>
        [DataMember]
        public double[][] SearchSpace;

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
        /// Specifies how accurate should be the result if the target value is given
        /// </summary>
        [DataMember]
        public double Epsilon;

        /// <summary>
        /// Number of dimensions
        /// </summary>
        [DataMember]
        public int Dimensions;
    }
}