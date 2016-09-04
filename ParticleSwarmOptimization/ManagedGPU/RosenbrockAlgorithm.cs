using System;
using ManagedCuda;

namespace ManagedGPU
{
    class RosenbrockAlgorithm : GenericCudaAlgorithm
    {
        protected CudaDeviceVariable<double> MinusOnes; 
        protected CudaDeviceVariable<double> Xopt;

        protected double Fopt;
        protected double Factor;

        public override void Dispose()
        {
            MinusOnes.Dispose();
            Xopt.Dispose();
            base.Dispose();
        }

        public RosenbrockAlgorithm(CudaParams parameters, StateProxy proxy) : base(parameters, proxy) { }

        protected override void Init()
        {
            var kernelFileName = KernelFile;
            var initKernel = Ctx.LoadKernel(kernelFileName, "generateData");
            Xopt = new CudaDeviceVariable<double>(DimensionsCount);

            var d_fopt = new CudaDeviceVariable<double>(1);

            long rseed = FunctionNumber + 10000 * InstanceNumber;

            double[] host_minus_ones = new double[DimensionsCount];

            for (int i = 0; i < DimensionsCount; i++)
            {
                host_minus_ones[i] = -1;
            }

            MinusOnes = host_minus_ones;

            Factor = Math.Max(1.0, Math.Sqrt(DimensionsCount)/8.0);

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
            get { return "f8_rosenbrock_kernel.ptx"; }
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
                    MinusOnes.DevicePointer,
                    Factor
             );  
        }
    }
}
