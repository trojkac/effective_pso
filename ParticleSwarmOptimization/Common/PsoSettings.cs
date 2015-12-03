using System;

namespace Common
{
    /// <summary>
    /// Computes fitness function
    /// </summary>
    /// <param name="values">fitness function argument which is N dimensional vector</param>
    /// <returns></returns>
    public delegate double FitnessFunction(double[] values); 

    public struct PsoSettings
    {
        /// <summary>
        /// Collection specifying types of particles and number of particles of each type
        /// </summary>
        public Tuple<PsoParticleType, int>[] Particles;
        
        /// <summary>
        /// Nx2 array where N is dimension of the search space.
        /// In this array minimum and maximum of each dimension are stored.
        /// </summary>
        public double[,] SearchSpace;

        public bool IterationsLimitCondition;
        /// <summary>
        /// Iterations limit.
        /// </summary>
        public int Iterations;

        public bool TargetValueCondition;
        /// <summary>
        /// If result of the optimization problem is known it can be used as a stop condition
        /// </summary>
        public double TargetValue;
        /// <summary>
        /// Specifies how accurate should be the result if the target value is given
        /// </summary>
        public double Epsilon;



    }
}