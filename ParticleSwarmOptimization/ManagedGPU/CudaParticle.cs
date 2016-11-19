using System;
using System.Linq;
using Algorithm;
using Common;

namespace ManagedGPU
{
    public class CudaParticle : Particle
    {
        private readonly StateProxy _proxy;

        internal CudaParticle(StateProxy proxy)
        {
            _proxy = proxy;
        }

        private double[] GetClampedLocation(double[] vector)
        {
            if (vector == null) return vector;
            return vector.Select((x, i) => Math.Min(Math.Max(x, -5.0), 5.0)).ToArray();
        }

        public override void UpdateNeighborhood(IParticle[] allParticles)
        {
            Neighborhood = allParticles.Where(particle => particle.Id != Id).ToArray();
            PushCpuState();
        }

        private void PushCpuState()
        {
            _proxy.CpuState = BestNeighborState();
        }

        private ParticleState BestNeighborState()
        {
            ParticleState best = Neighborhood[0].PersonalBest;

            foreach (var particle in Neighborhood)
            {
                if (Optimization.IsBetter(particle.PersonalBest.FitnessValue, best.FitnessValue) < 0)
                    best = particle.PersonalBest;
            }

            return best;
        }

        public void Init()
        {
            Init(new ParticleState(null, null), null, null);
        }

        public override void Init(ParticleState state, double[] velocity, DimensionBound[] bounds = null)
        {
            CurrentState = _proxy.GpuState;
        }

        public override int Id
        {
            get { return _id; }
        }

        public override void UpdateVelocity(IState<double[], double[]> globalBest) { }

        private void PullGpuState()
        {
            CurrentState = _proxy.GpuState;
        }

        public override void Transpose(IFitnessFunction<double[], double[]> function)
        {
            PullGpuState();
            var location = GetClampedLocation(CurrentState.Location);
            CurrentState = new ParticleState(location, function.Evaluate(location));

            if (PersonalBest.FitnessValue == null || CurrentIsBetterThanBest())
                PersonalBest = CurrentState;
        }

        private bool CurrentIsBetterThanBest()
        {
            return Optimization.IsBetter(CurrentState.FitnessValue, PersonalBest.FitnessValue) < 0;
        }
    }
}
