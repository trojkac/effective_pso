using Common;

namespace ManagedGPU
{
    internal class StateProxy
    {
        internal StateProxy(CudaParams parameters)
        {
            CpuState = ParticleStateFactory.Create(parameters.LocationDimensions, 1);
            CpuState.Location = RandomGenerator.GetInstance().RandomVector(parameters.LocationDimensions); 
            GpuState = ParticleStateFactory.Create(parameters.LocationDimensions, 1);
            GpuState.Location = RandomGenerator.GetInstance().RandomVector(parameters.LocationDimensions);
        }

        public ParticleState CpuState { get; set; }

        public ParticleState GpuState { get; set; }
    }
}
