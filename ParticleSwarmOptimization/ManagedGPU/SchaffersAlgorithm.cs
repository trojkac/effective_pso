﻿using ManagedCuda;

namespace ManagedGPU
{
    class SchaffersAlgorithm : GenericCudaAlgorithm
    {
        protected CudaDeviceVariable<double> M;
        protected CudaDeviceVariable<double> B;
        protected CudaDeviceVariable<double> Xopt;



        public double Conditioning { get; set; }

        public bool IllformedSeed { get; set; }

        protected double Penalty = 10.0;
        protected double Asymmetric = 0.5;

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

        public SchaffersAlgorithm(CudaParams parameters, StateProxy proxy) : base(parameters, proxy) { }

        protected override void Init()
        {
            var kernelFileName = KernelFile;
            var initKernel = Ctx.LoadKernel(kernelFileName, "generateData");
            B = new CudaDeviceVariable<double>(DimensionsCount);
            M = new CudaDeviceVariable<double>(DimensionsCount * DimensionsCount);
            Xopt = new CudaDeviceVariable<double>(DimensionsCount);

            var d_fopt = new CudaDeviceVariable<double>(1);

            int seedBase;

            if (IllformedSeed)
                seedBase = 17;
            else
                seedBase = FunctionNumber;

            long rseed = seedBase + 10000 * InstanceNumber;

            initKernel.Run(
                DimensionsCount, 
                rseed, 
                FunctionNumber, 
                InstanceNumber, 
                M.DevicePointer,
                B.DevicePointer, 
                Xopt.DevicePointer,
                d_fopt.DevicePointer,
                Conditioning);

            double[] fopt_arr = d_fopt;

            Fopt = fopt_arr[0];
        }

        protected override string KernelFile
        {
            get { return "f17_schaffers_kernel.ptx"; }
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
                    Asymmetric,
                    Penalty
             );  
        }
    }
}
