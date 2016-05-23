using System;
using Common;

namespace Algorithm
{
    public class ParticleFactory
    {
        public const double MinInitialVelocity = -2;
        public const double MaxInitialVelocity = 2;
        private static readonly IMetric<double[]> Metric = new Euclidean();
        private static readonly IOptimization<double[]> Optimization = new FirstValueOptimization(); 


        public static IParticle Create(PsoParticleType type, int locationDim, int fitnessDim, IFitnessFunction<double[],double[]> function, Tuple<double, double>[] bounds = null)
        {
            IParticle particle = null;
            var rand = RandomGenerator.GetInstance();
            switch (type)
            {
                case PsoParticleType.Standard:
                case PsoParticleType.FullyInformed:
                    particle = new StandardParticle(Optimization,Metric);
                    break;
            }

            var x = bounds != null ? rand.RandomVector(locationDim,bounds) : rand.RandomVector(locationDim);
            particle.Init(new ParticleState(x,function.Evaluate(x)), 
                rand.RandomVector(locationDim, MinInitialVelocity, MaxInitialVelocity), bounds);
            return particle;
        } 
    }
}