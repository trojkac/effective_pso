using System;
using Common;

namespace Algorithm
{
    public class ParticleFactory
    {
        public const double MinInitialVelocity = -2;
        public const double MaxInitialVelocity = 2;

        public static IParticle Create(PsoParticleType type, 
            int locationDim, 
            int fitnessDim,
            IFitnessFunction<double[],double[]> function, 
            double restartEpsilon=double.MaxValue,
            int iterationsToRestart=int.MaxValue,
            DimensionBound[] bounds = null, 
            double[] initVelocity = null
            )
        {
            IParticle particle = null;
            var rand = RandomGenerator.GetInstance();
            switch (type)
            {
                case PsoParticleType.Standard:
                case PsoParticleType.FullyInformed:
                    particle = new StandardParticle(restartEpsilon, iterationsToRestart);
                    break;
            }

            var x = bounds != null ? rand.RandomVector(locationDim,bounds) : rand.RandomVector(locationDim);
            particle.Init(new ParticleState(x,function.Evaluate(x)), 
                initVelocity ?? rand.RandomVector(locationDim, MinInitialVelocity, MaxInitialVelocity), bounds);
            return particle;
        } 
    }
}