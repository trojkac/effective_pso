using Algorithm;
using Common;

namespace ManagedGPU
{
    public class CudaParticle : Particle
    {
        private CudaAlgorithm _algorithm;

        public CudaParticle(CudaAlgorithm alg)
        {
            _algorithm = alg;
        }

        public override void UpdateNeighborhood(IParticle[] allParticles) { }

        public override void Init(ParticleState state, double[] velocity)
        {
            throw new System.NotImplementedException();
        }

        public override int Id
        {
            get { return _id; }
        }

        public override void UpdateVelocity() { }

        public override void Translate() { }

        public override void UpdatePersonalBest(IFitnessFunction<double[], double[]> function)
        {
            throw new System.NotImplementedException();
        }
    }
}
