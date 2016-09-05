using ManagedCuda;

namespace ManagedGPU
{
    class GallagherAlgorithm : GenericCudaAlgorithm
    {
        public double PeaksCount { get; set; }

        protected CudaDeviceVariable<double> Rotation;
        protected CudaDeviceVariable<double> PeakValues;
        protected CudaDeviceVariable<double> XLocal;
        protected CudaDeviceVariable<double> ArrScales;

        public override void Dispose()
        {
            Rotation.Dispose();
            PeakValues.Dispose();
            XLocal.Dispose();
            ArrScales.Dispose();
            base.Dispose();
        }

        protected override void Cleanup()
        {
            Dispose();
        }

        public GallagherAlgorithm(CudaParams parameters, StateProxy proxy) : base(parameters, proxy) { }

        protected override void Init()
        {
            var kernelFileName = KernelFile;
            var initKernel = Ctx.LoadKernel(kernelFileName, "generateData");

            Rotation = new CudaDeviceVariable<double>(DimensionsCount * DimensionsCount);
            PeakValues = new CudaDeviceVariable<double>((int)PeaksCount);
            XLocal = new CudaDeviceVariable<double>(DimensionsCount * (int)PeaksCount);
            ArrScales = new CudaDeviceVariable<double>(DimensionsCount * (int)PeaksCount);

            long rseed = FunctionNumber + 10000 * InstanceNumber;

            initKernel.Run(
                DimensionsCount, 
                rseed, 
                Rotation.DevicePointer,
                PeaksCount,
                PeakValues.DevicePointer,
                XLocal.DevicePointer,
                ArrScales.DevicePointer);
        }

        protected override string KernelFile
        {
            get { return "f21_gallagher_kernel.ptx"; }
        }

        protected override void RunUpdateVelocityKernel()
        {
            UpdateVelocity.Run(
                    DevicePositions.DevicePointer,
                    DeviceVelocities.DevicePointer,
                    DevicePersonalBests.DevicePointer,
                    DevicePersonalBestValues.DevicePointer,
                    DeviceNeighbors.DevicePointer,
                    ParticlesCount,
                    DimensionsCount,
                    Random(Rng),
                    Random(Rng)
                );
        }

        protected override void RunTransposeKernel()
        {
            Transpose.Run(
                    DevicePositions.DevicePointer,
                    DeviceVelocities.DevicePointer,
                    DevicePersonalBests.DevicePointer,
                    DevicePersonalBestValues.DevicePointer,
                    ParticlesCount,
                    DimensionsCount,
                    Rotation.DevicePointer,
                    PeaksCount,
                    PeakValues.DevicePointer,
                    XLocal.DevicePointer,
                    ArrScales.DevicePointer
             );  
        }
    }
}
