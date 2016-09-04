using Common;

namespace ManagedGPU
{
    public class StateProxy
    {
        internal StateProxy(CudaParams parameters)
        {
            var rand = RandomGenerator.GetInstance();
            var x = parameters.Bounds != null ? rand.RandomVector(parameters.LocationDimensions, parameters.Bounds) : rand.RandomVector(parameters.LocationDimensions);
        
            CpuState = new ParticleState(x, parameters.FitnessFunction.Evaluate(x));
            GpuState = new ParticleState(x, parameters.FitnessFunction.Evaluate(x));
        }

        public ParticleState CpuState { get; set; }

        public ParticleState GpuState { get; set; }
    }
}
