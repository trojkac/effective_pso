using Common;

namespace Algorithm
{
    public class ParticleFactory
    {
        public const double MinInitialVelocity = -2;
        public const double MaxInitialVelocity = 2;

        public static IParticle Create(PsoParticleType type,int locationDim,int fitnessDim)
        {
            IParticle particle = null;
            switch (type)
            {
                case PsoParticleType.Standard:
                case PsoParticleType.FullyInformed:
                    particle = new StandardParticle();
                    break;
            }
            particle.Init(ParticleStateFactory.Create(locationDim, fitnessDim),
                RandomGenerator.GetInstance().RandomVector(locationDim, MinInitialVelocity, MaxInitialVelocity));
            return particle;
        } 
    }
}