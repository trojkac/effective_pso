using ManagedCuda;

namespace ManagedGPU
{
    class LunacekBiRastriginAlgorithm : GenericCudaAlgorithm
    {
        protected CudaDeviceVariable<double> Rotation1;
        protected CudaDeviceVariable<double> Rotation2;
        protected CudaDeviceVariable<double> Xopt;

        public override void Dispose()
        {
            Rotation1.Dispose();
            Rotation2.Dispose();
            Xopt.Dispose();
            base.Dispose();
        }

        public LunacekBiRastriginAlgorithm(CudaParams parameters, StateProxy proxy) : base(parameters, proxy) { }

        protected override void Init()
        {
            var kernelFileName = KernelFile;
            var initKernel = Ctx.LoadKernel(kernelFileName, "generateData");
            Rotation2 = new CudaDeviceVariable<double>(DimensionsCount * DimensionsCount);
            Rotation1 = new CudaDeviceVariable<double>(DimensionsCount * DimensionsCount);
            Xopt = new CudaDeviceVariable<double>(DimensionsCount);

            long rseed = FunctionNumber + 10000 * InstanceNumber;

            initKernel.Run(
                DimensionsCount, 
                rseed, 
                Rotation1.DevicePointer, 
                Rotation2.DevicePointer, 
                Xopt.DevicePointer);
        }

        protected override string KernelFile
        {
            get { return "f24_lunacek_bi_rastrigin_kernel.ptx"; }
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
                    Xopt.DevicePointer,
                    Rotation1.DevicePointer,
                    Rotation2.DevicePointer
             );  
        }
    }
}
