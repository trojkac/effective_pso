using System;
using ManagedCuda;

namespace ManagedGPU
{
    class GallagherAlgorithm : GenericCudaAlgorithm
    {
        public int PeaksCount { get; set; }

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
            initKernel.BlockDimensions = 1;
            initKernel.GridDimensions = 1;

            Rotation = new CudaDeviceVariable<double>(DimensionsCount * DimensionsCount);
            PeakValues = new CudaDeviceVariable<double>(PeaksCount);
            XLocal = new CudaDeviceVariable<double>(DimensionsCount * PeaksCount);
            ArrScales = new CudaDeviceVariable<double>(DimensionsCount * PeaksCount);

            int rseed = FunctionNumber + 10000 * InstanceNumber;
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
