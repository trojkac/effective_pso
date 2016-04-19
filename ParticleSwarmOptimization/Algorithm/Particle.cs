using Common;

namespace Algorithm
{
    public abstract class Particle : IParticle
    {

        private static int _idCounter;
        protected int _id;
    
        protected Particle() {
            _id = ++_idCounter;
        }

        protected IParticle[] Neighborhood;
        public abstract void Init(ParticleState state, double[] velocity);
        public abstract void UpdateVelocity();
        public abstract void UpdatePersonalBest(IFitnessFunction<double[], double[]> function);
        public abstract void UpdateNeighborhood(IParticle[] allParticles); 
        public ParticleState PersonalBest { get; protected set; }
        public ParticleState CurrentState { get; protected set; }
        public double[] Velocity { get; protected set; }
        public abstract int Id { get; }
        public abstract void Translate();
    }
}