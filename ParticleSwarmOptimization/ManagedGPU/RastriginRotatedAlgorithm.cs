using ManagedCuda;

namespace ManagedGPU
{
    class RastriginRotatedAlgorithm : GenericCudaAlgorithm
    {
        protected CudaDeviceVariable<double> Xopt;
        protected CudaDeviceVariable<double> M;
        protected CudaDeviceVariable<double> B;

        protected double Fopt;

        protected double Asymmetric = 0.2;

        public override void Dispose()
        {
            Xopt.Dispose();
            M.Dispose();
            B.Dispose();
            base.Dispose();
        }

        public RastriginRotatedAlgorithm(CudaParams parameters, StateProxy proxy) : base(parameters, proxy) { }

        protected override void Init()
        {
            var kernelFileName = KernelFile;
            var initKernel = Ctx.LoadKernel(kernelFileName, "generateData");
            Xopt = new CudaDeviceVariable<double>(DimensionsCount);

            var d_fopt = new CudaDeviceVariable<double>(1);

            long rseed = FunctionNumber + 10000 * InstanceNumber;

            initKernel.Run(
                DimensionsCount, 
                rseed, 
                FunctionNumber, 
                InstanceNumber,
                M.DevicePointer,
                B.DevicePointer,
                Xopt.DevicePointer,
                d_fopt.DevicePointer);

            double[] fopt_arr = d_fopt;

            Fopt = fopt_arr[0];
        }

        protected override string KernelFile
        {
            get { return "f15_rastrigin_rotated_kernel.ptx"; }
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
                    Asymmetric,
                    M.DevicePointer,
                    B.DevicePointer
             );  
        }
    }
}
