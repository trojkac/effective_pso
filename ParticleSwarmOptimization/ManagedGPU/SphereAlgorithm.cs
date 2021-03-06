﻿using System;
using ManagedCuda;

namespace ManagedGPU
{
    class SphereAlgorithm : GenericCudaAlgorithm
    {
        protected CudaDeviceVariable<double> Xopt;

        public SphereAlgorithm(CudaParams parameters, StateProxy proxy) : base(parameters, proxy) { }

        protected override void Init()
        {
            var kernelFileName = KernelFile;
            var initKernel = Ctx.LoadKernel(kernelFileName, "generateData");
            Xopt = new CudaDeviceVariable<double>(DimensionsCount);

            var d_fopt = new CudaDeviceVariable<double>(1);

            int rseed = FunctionNumber + 10000 * InstanceNumber;

            initKernel.Run(
                DimensionsCount, 
                rseed, 
                FunctionNumber, 
                InstanceNumber, 
                Xopt.DevicePointer, 
                d_fopt.DevicePointer);

            double[] fopt_arr = d_fopt;

            d_fopt.Dispose();

            Fopt = fopt_arr[0];
        }



        protected override string KernelFile
        {
            get { return "f1_sphere_kernel.ptx"; }
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
                 Fopt
             );  
        }

        public override void Dispose()
        {
            Xopt.Dispose();
            base.Dispose();
        }

        protected override void Cleanup()
        {
            Dispose();
        }
    }
}
