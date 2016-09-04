using ManagedCuda;

namespace ManagedGPU
{
    class BuecheRastriginAlgorithm : GenericCudaAlgorithm
    {
        protected CudaDeviceVariable<double> Xopt;

        protected double Fopt;
        protected double Penalty = 100.0;

        public override void Dispose()
        {
            Xopt.Dispose();
            base.Dispose();
        }

        public BuecheRastriginAlgorithm(CudaParams parameters, StateProxy proxy) : base(parameters, proxy) { }

        protected override void Init()
        {
            var kernelFileName = KernelFile;
            var initKernel = Ctx.LoadKernel(kernelFileName, "generateData");
            Xopt = new CudaDeviceVariable<double>(DimensionsCount);

            var d_fopt = new CudaDeviceVariable<double>(1);

            long rseed = (long)(3 + 10000 * InstanceNumber);

            initKernel.Run(
                DimensionsCount, 
                rseed, 
                FunctionNumber, 
                InstanceNumber,
                Xopt.DevicePointer, 
                d_fopt.DevicePointer);

            double[] fopt_arr = d_fopt;

            Fopt = fopt_arr[0];
        }
        protected override string KernelFile
        {
            get { return "f4_buche_rastrigin_kernel.ptx"; }
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
                    Fopt,
                    Penalty
             );  
        }
    }
}
