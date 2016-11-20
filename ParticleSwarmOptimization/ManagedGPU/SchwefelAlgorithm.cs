using ManagedCuda;

namespace ManagedGPU
{
    class SchwefelAlgorithm : GenericCudaAlgorithm
    {
        protected CudaDeviceVariable<double> M;
        protected CudaDeviceVariable<double> B;
        protected CudaDeviceVariable<double> Xopt;

        protected double Fopt;

        protected CudaDeviceVariable<double> Tmp1;
        protected CudaDeviceVariable<double> Tmp2;

        public override void Dispose()
        {
            M.Dispose();
            B.Dispose();
            Xopt.Dispose();
            Tmp1.Dispose();
            Tmp2.Dispose();
            base.Dispose();
        }

        protected override void Cleanup()
        {
            Dispose();
        }

        private double rseed;

        public SchwefelAlgorithm(CudaParams parameters, StateProxy proxy) : base(parameters, proxy) { }

        protected override void Init()
        {
            var kernelFileName = KernelFile;
            var initKernel = Ctx.LoadKernel(kernelFileName, "generateData");
            B = new CudaDeviceVariable<double>(DimensionsCount);
            M = new CudaDeviceVariable<double>(DimensionsCount * DimensionsCount);
            Xopt = new CudaDeviceVariable<double>(DimensionsCount);
            Tmp1 = new CudaDeviceVariable<double>(DimensionsCount);
            Tmp2 = new CudaDeviceVariable<double>(DimensionsCount);

            var d_fopt = new CudaDeviceVariable<double>(1);

            rseed = FunctionNumber + 10000 * InstanceNumber;

            initKernel.Run(
                DimensionsCount, 
                rseed, 
                FunctionNumber, 
                InstanceNumber, 
                M.DevicePointer,
                B.DevicePointer,
                Xopt.DevicePointer,
                d_fopt.DevicePointer,
                Tmp1.DevicePointer,
                Tmp2.DevicePointer);

            double[] fopt_arr = d_fopt;

            Fopt = fopt_arr[0];
        }

        protected override string KernelFile
        {
            get { return "f20_schwefel_kernel.ptx"; }
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
                    M.DevicePointer,
                    B.DevicePointer,
                    Fopt,
                    Tmp1.DevicePointer,
                    Tmp2.DevicePointer,
                    rseed
             );  
        }
    }
}
