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
        /// Specifies how accurate should the result be if the target value is given
        /// </summary>
        [DataMember]
        public double Epsilon;

        /// <summary>
        /// Number of dimensions
        /// </summary>
        [DataMember]
        public int Dimensions;

        public PsoSettings(UserPsoParameters psoParams, UserFunctionParameters functionParams)
        {
            Epsilon = 0;
            TargetValue = 0;
            Iterations = 0;
            FitnessFunction = null;

            switch (functionParams.FitnessFunctionType)
            {
                case FitnessFunctionType.Quadratic:
                    {
                        AbstractFitnessFunction function = new QuadraticFunction(functionParams);
                        FitnessFunction = function.Calculate;
                        goto default;
                    }
                case FitnessFunctionType.Rastrigin:
                    {
                        AbstractFitnessFunction function = new RastriginFunction(functionParams);
                        FitnessFunction = function.Calculate;
                        goto default;
                    }
                case FitnessFunctionType.Rosenbrock:
                    {
                        AbstractFitnessFunction function = new RosenbrockFunction(functionParams);
                        FitnessFunction = function.Calculate;
                        goto default;
                    }
                default:
                    {
                        Particles = new Tuple<PsoParticleType, int>[2];
                        Particles[0] = new Tuple<PsoParticleType, int>(PsoParticleType.Standard, psoParams.StandardParticles);
                        Particles[1] = new Tuple<PsoParticleType, int>(PsoParticleType.FullyInformed, psoParams.FullyInformedParticles);

                        SearchSpace = new double[functionParams.Dimension][];
                        for (int i = 0; i < functionParams.Dimension; i++)
                        {
                            SearchSpace[i] = new double[2];
                            SearchSpace[i][0] = functionParams.SearchSpace[i].Item1;
                            SearchSpace[i][1] = functionParams.SearchSpace[i].Item2;
                        }

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
                        break;
                    }
            }
        }
    }
}