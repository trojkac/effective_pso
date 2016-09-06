using System;
using ManagedCuda;

namespace ManagedGPU
{
    class WeierstrassAlgorithm : GenericCudaAlgorithm
    {
        protected CudaDeviceVariable<double> Xopt;
        protected CudaDeviceVariable<double> M;
        protected CudaDeviceVariable<double> B;

        protected double Fopt;

        protected double Penalty = 10.0;

        protected double F0 = 0.0;
        protected CudaDeviceVariable<double> Ak;
        protected CudaDeviceVariable<double> Bk;

        public override void Dispose()
        {
            Xopt.Dispose();
            M.Dispose();
            B.Dispose();
            Ak.Dispose();
            Bk.Dispose();
            base.Dispose();
        }

        protected override void Cleanup()
        {
            Dispose();
        }

        public WeierstrassAlgorithm(CudaParams parameters, StateProxy proxy) : base(parameters, proxy) { }

        protected override void Init()
        {
            var kernelFileName = KernelFile;
            var initKernel = Ctx.LoadKernel(kernelFileName, "generateData");
            Xopt = new CudaDeviceVariable<double>(DimensionsCount);
            M = new CudaDeviceVariable<double>(DimensionsCount * DimensionsCount);
            B = new CudaDeviceVariable<double>(DimensionsCount);

            var d_fopt = new CudaDeviceVariable<double>(1);

            long rseed = FunctionNumber + 10000 * InstanceNumber;

            double[] h_ak = new double[12];
            double[] h_bk = new double[12];

            for (int i = 0; i < 12; i++)
            {
                h_ak[i] = Math.Pow(0.5, i);
                h_bk[i] = Math.Pow(3.0, i);
                F0 += h_ak[i]*Math.Cos(Math.PI*h_bk[i]);
            }

            Ak = h_ak;
            Bk = h_bk;

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
            get { return "f16_weierstrass_kernel.ptx"; }
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
                    Penalty,
                    M.DevicePointer,
                    B.DevicePointer,
                    Ak.DevicePointer,
                    Bk.DevicePointer,
                    F0
             );  
        }
    }
}
