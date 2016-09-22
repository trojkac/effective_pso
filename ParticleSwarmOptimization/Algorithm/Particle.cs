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
        public abstract void Init(ParticleState state, double[] velocity, DimensionBound[] bounds = null);
        public void InitializeVelocity(IParticle particle)
        {
            if (Velocity == null)
            {
                return;
            }
            for (var i = 0; i < Velocity.Length; i++)
            {
                Velocity[i] = (particle.CurrentState.Location[i] - CurrentState.Location[i]) / 2;
            }
        }
        public abstract void UpdateVelocity(IState<double[], double[]> globalBest);
        public abstract void Transpose(IFitnessFunction<double[], double[]> function);
        public abstract void UpdateNeighborhood(IParticle[] allParticles);
        public virtual ParticleState PersonalBest { get; protected set; }
        public virtual ParticleState CurrentState { get; protected set; }
        public DimensionBound[] Bounds;

        public double[] Velocity { get; protected set; }
        public abstract int Id { get; }
        public string ToLog()
        {
            var sb = new StringBuilder();
            sb.Append(Id).Append(": ");
            foreach (var d in CurrentState.Location)
            {
                sb.Append(d);
                sb.Append(" ");
            }
            return sb.ToString();
        }
    }
}