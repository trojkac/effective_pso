﻿using System;
using ManagedCuda;

namespace ManagedGPU
{
    class RosenbrockRotatedAlgorithm : GenericCudaAlgorithm
    {
        protected CudaDeviceVariable<double> M;
        protected CudaDeviceVariable<double> B;



        public override void Dispose()
        {
            M.Dispose();
            B.Dispose();
            base.Dispose();
        }

        protected override void Cleanup()
        {
            Dispose();
        }

        public RosenbrockRotatedAlgorithm(CudaParams parameters, StateProxy proxy) : base(parameters, proxy) { }

        protected override void Init()
        {
            var kernelFileName = KernelFile;
            var initKernel = Ctx.LoadKernel(kernelFileName, "generateData");
            B = new CudaDeviceVariable<double>(DimensionsCount);
            M = new CudaDeviceVariable<double>(DimensionsCount * DimensionsCount);

            var d_fopt = new CudaDeviceVariable<double>(1);

            long rseed = FunctionNumber + 10000 * InstanceNumber;

            initKernel.Run(
                DimensionsCount, 
                rseed, 
                FunctionNumber, 
                InstanceNumber, 
                M.DevicePointer, 
                B.DevicePointer,
                d_fopt.DevicePointer);

            double[] fopt_arr = d_fopt;

            Fopt = fopt_arr[0];
        }

        protected override string KernelFile
        {
            get { return "f9_rosenbrock_rotated_kernel.ptx"; }
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
                    M.DevicePointer,
                    B.DevicePointer,
                    Fopt
             );  
        }
    }
}
