using System;
using System.CodeDom;
using System.Text;
using Common;

namespace Algorithm
{
    public abstract class Particle : IParticle, ILogable
    {

        private static int _idCounter;
        protected int _id;
    
        protected Particle() {
            _id = ++_idCounter;
        }

        protected IParticle[] Neighborhood;
        public abstract void Init(ParticleState state, double[] velocity, Tuple<double, double>[] bounds = null);
        public abstract void UpdateVelocity();
        public abstract void UpdatePersonalBest(IFitnessFunction<double[], double[]> function);
        public abstract void UpdateNeighborhood(IParticle[] allParticles); 
        public ParticleState PersonalBest { get; protected set; }
        public ParticleState CurrentState { get; protected set; }
        public Tuple<double, double>[] Bounds;
        public double[] Velocity { get; protected set; }
        public abstract int Id { get; }
        public abstract void Translate();
        public string ToLog()
        {
            var sb = new StringBuilder();
            sb.Append("L: ");
            foreach (var d in CurrentState.Location)
            {
                sb.Append(d);
                sb.Append(" ");
            }
            return sb.ToString();
        }
    }
}