using System;
using System.Linq;
using System.Text;
using Common;

namespace Algorithm
{
    public abstract class Particle : IParticle, ILogable
    {
        private static int _idCounter;
        private readonly int _iterationsToRestart;
        protected int _id;
        private int _sinceLastImprovement;
        public DimensionBound[] Bounds;
        protected IMetric<double[]> Metric;
        protected IParticle[] Neighborhood;

        protected Particle(double restartEpsilon = 0.0, int iterationsToRestart = int.MaxValue)
        {
            _id = ++_idCounter;
            Metric = PsoServiceLocator.Instance.GetService<IMetric<double[]>>();
            Optimization = PsoServiceLocator.Instance.GetService<IOptimization<double[]>>();

            _iterationsToRestart = iterationsToRestart;
        }

        protected IOptimization<double[]> Optimization { get; set; }


        public double[] Velocity { get; protected set; }

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

        public void InitializeVelocity(IParticle particle)
        {
            if (Velocity == null)
            {
                return;
            }
            for (var i = 0; i < Velocity.Length; i++)
            {
                Velocity[i] = (particle.CurrentState.Location[i] - CurrentState.Location[i])/2;
            }
        }

        public abstract void UpdateVelocity(IState<double[], double[]> globalBest);
        public abstract void UpdateNeighborhood(IParticle[] allParticles);
        public virtual ParticleState PersonalBest { get; protected set; }
        public virtual ParticleState CurrentState { get; protected set; }
        public abstract int Id { get; }

        public virtual void Init(ParticleState particleState, double[] velocity, DimensionBound[] bounds = null)
        {
            CurrentState = particleState;
            PersonalBest = particleState;
            Velocity = velocity;
            Bounds = bounds;
        }

        public virtual void Transpose(IFitnessFunction<double[], double[]> function)
        {
            double[] newLocation;
            var restart = _sinceLastImprovement == _iterationsToRestart;
            if (restart)
            {
                newLocation = RandomGenerator.GetInstance().RandomVector(CurrentState.Location.Length, Bounds);
                _sinceLastImprovement = 0;
            }
            else
            {
                newLocation = GetClampedLocation(CurrentState.Location.Select((x, i) => x + Velocity[i]).ToArray());
            }
            var newVal = function.Evaluate(newLocation);
            var oldBest = PersonalBest;
            CurrentState = new ParticleState(newLocation, newVal);

            if (
                Optimization.IsBetter(newVal, PersonalBest.FitnessValue) < 0)
            {
                PersonalBest = CurrentState;
                _sinceLastImprovement = 0;
            }
            else
            {
                _sinceLastImprovement++;
            }
        }

        protected double[] GetClampedLocation(double[] vector)
        {
            if (Bounds == null || vector == null) return vector;
            return vector.Select((x, i) => Math.Min(Math.Max(x, Bounds[i].Min), Bounds[i].Max)).ToArray();
        }
    }
}