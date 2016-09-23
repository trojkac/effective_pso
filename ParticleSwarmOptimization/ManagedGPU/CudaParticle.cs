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
            CurrentState = proxy.CpuState;
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
            ParticleState best = Neighborhood[0].CurrentState;

            foreach (var particle in Neighborhood)
            {
                if (Optimization.IsBetter(particle.CurrentState.FitnessValue, best.FitnessValue) < 0)
                    best = particle.CurrentState;
            }

            return best;
        }

        public override void Init(ParticleState state, double[] velocity, DimensionBound[] bounds = null)
        {
            CurrentState = state;
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
            CurrentState = new ParticleState(CurrentState.Location, function.Evaluate(CurrentState.Location));

            if (PersonalBest.FitnessValue == null || CurrentIsBetterThanBest())
                PersonalBest = CurrentState;
        }

        private bool CurrentIsBetterThanBest()
        {
            return Optimization.IsBetter(CurrentState.FitnessValue, PersonalBest.FitnessValue) < 0;
        }
    }
}
