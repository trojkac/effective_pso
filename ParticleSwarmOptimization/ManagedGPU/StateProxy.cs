using Common;

namespace ManagedGPU
{
    internal class StateProxy
    {
        internal StateProxy(CudaParams parameters)
        {
            CpuState = ParticleStateFactory.Create(parameters.Dimensions, 1);
            CpuState.Location = RandomGenerator.GetInstance().RandomVector(parameters.Dimensions); 
            GpuState = ParticleStateFactory.Create(parameters.Dimensions, 1);
            GpuState.Location = RandomGenerator.GetInstance().RandomVector(parameters.Dimensions);
        }

        public ParticleState CpuState { get; set; }

        public ParticleState GpuState { get; set; }
    }
}
