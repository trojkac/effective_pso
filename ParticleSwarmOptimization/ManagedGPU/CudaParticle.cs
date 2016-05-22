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
            _proxy.CpuState = (ParticleState)BestNeighborState().Clone();
        }

        private ParticleState BestNeighborState()
        {
            ParticleState best = null;

            foreach (var particle in Neighborhood)
            {
                if (best == null)
                    best = particle.CurrentState;
                else if (particle.CurrentState.IsBetter(best))
                    best = particle.CurrentState;
            }

            return best;
        }

        public override void Init(ParticleState state, double[] velocity, Tuple<double, double>[] bounds = null)
        {
            CurrentState = state;
        }

        public override int Id
        {
            get { return _id; }
        }

        public override void UpdateVelocity() { }

        public override void Translate()
        {
            PullGpuState();
        }

        private void PullGpuState()
        {
            var gpustate = (ParticleState) _proxy.GpuState.Clone();
            CurrentState = gpustate;
        }

        public override void UpdatePersonalBest(IFitnessFunction<double[], double[]> function)
        {
            CurrentState.FitnessValue = function.Evaluate(CurrentState.Location);
            PersonalBest = PersonalBest == null || CurrentState.IsBetter(PersonalBest) ? (ParticleState)CurrentState.Clone() : PersonalBest;
        }
    }
}
