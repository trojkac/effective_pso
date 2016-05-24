namespace ManagedGPU
{
    public class CudaParams
    {
        public int ParticlesCount { get; set; }

        public int LocationDimensions { get; set; }

        public int FitnessDimensions { get; set; }

        public int Iterations { get; set; }

        public bool SyncWithCpu { get; set; }

        //Fitness function

        //Bounds
    }
}
