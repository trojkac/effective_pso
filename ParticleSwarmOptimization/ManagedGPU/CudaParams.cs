using Common;

namespace ManagedGPU
{
    public class CudaParams
    {
        public int FunctionNumber { get; set; }

        public int InstanceNumber { get; set; }

        public int ParticlesCount { get; set; }

        public int LocationDimensions { get; set; }

        public int FitnessDimensions { get; set; }

        public int Iterations { get; set; }

        public bool SyncWithCpu { get; set; }

        public DimensionBound[] Bounds { get; set; }

        public IFitnessFunction<double[], double[]> FitnessFunction { get; set; }

        //Bounds
    }
}
