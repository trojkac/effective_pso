using Common;

namespace ManagedGPU
{
    internal class StateProxy
    {
        internal StateProxy(CudaParams parameters)
        {
            CpuState = new ParticleState(RandomGenerator.GetInstance().RandomVector(parameters.LocationDimensions), 
                RandomGenerator.GetInstance().RandomVector(parameters.FitnessDimensions));
            GpuState = new ParticleState(RandomGenerator.GetInstance().RandomVector(parameters.LocationDimensions), 
                RandomGenerator.GetInstance().RandomVector(parameters.FitnessDimensions));
        }

        public ParticleState CpuState { get; set; }

        public ParticleState GpuState { get; set; }
    }
}
