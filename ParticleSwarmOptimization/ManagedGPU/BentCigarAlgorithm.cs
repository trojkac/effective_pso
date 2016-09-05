using ManagedCuda;

namespace ManagedGPU
{
    class BentCigarAlgorithm : GenericCudaAlgorithm
    {

        protected CudaDeviceVariable<double> M;
        protected CudaDeviceVariable<double> B;
        protected CudaDeviceVariable<double> Xopt;

        protected double Fopt;
        protected double AsymmetricBeta = 0.5;

        public override void Dispose()
        {
            M.Dispose();
            B.Dispose();
            Xopt.Dispose();
            base.Dispose();
        }

        protected override void Cleanup()
        {
            Dispose();
        }

        public BentCigarAlgorithm(CudaParams parameters, StateProxy proxy) : base(parameters, proxy) { }

        protected override void Init()
        {
            var kernelFileName = KernelFile;
            var initKernel = Ctx.LoadKernel(kernelFileName, "generateData");
            B = new CudaDeviceVariable<double>(DimensionsCount);
            M = new CudaDeviceVariable<double>(DimensionsCount * DimensionsCount);
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
            get { return "f12_bent_cigar_kernel.ptx"; }
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
                    M.DevicePointer,
                    B.DevicePointer,
                    Fopt,
                    AsymmetricBeta
             );
        }
    }
}
