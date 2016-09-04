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

        public RosenbrockAlgorithm(CudaParams parameters, StateProxy proxy)
        {
            _proxy = proxy;
            _particlesCount = parameters.ParticlesCount;
            _dimensionsCount = parameters.LocationDimensions;
            _iterations = parameters.Iterations;
            _fitnessFunction = parameters.FitnessFunction;
            _syncWithCpu = parameters.SyncWithCpu;
            _functionNumber = parameters.FunctionNumber;
            _instanceNumber = parameters.InstanceNumber;
        }

        protected override void Init()
        {
            var kernelFileName = KernelFile;
            var initKernel = ctx.LoadKernel(kernelFileName, "generateData");
            Xopt = new CudaDeviceVariable<double>(_dimensionsCount);

            var d_fopt = new CudaDeviceVariable<double>(1);

            long rseed = _functionNumber + 10000 * _instanceNumber;

            double[] host_minus_ones = new double[_dimensionsCount];

            for (int i = 0; i < _dimensionsCount; i++)
            {
                host_minus_ones[i] = -1;
            }

            MinusOnes = host_minus_ones;

            Factor = Math.Max(1.0, Math.Sqrt(_dimensionsCount)/8.0);

            initKernel.Run(
                _dimensionsCount, 
                rseed, 
                _functionNumber, 
                _instanceNumber, 
                Xopt.DevicePointer,
                d_fopt.DevicePointer);

            double[] fopt_arr = d_fopt;

            Fopt = fopt_arr[0];
        }

        protected override string KernelFile
        {
            get { return "f8_rosenbrock_kernel.ptx"; }
        }

        protected override void RunUpdateParticleKernel()
        {
            _updateParticle.Run(
                    _devicePositions.DevicePointer,
                    _deviceVelocities.DevicePointer,
                    _devicePersonalBests.DevicePointer,
                    _deviceGlobalBests.DevicePointer,
                    _particlesCount,
                    _dimensionsCount,
                    Random(_rng),
                    Random(_rng)
                );
        }

        protected override void RunUpdatePersonalBestKernel()
        {
            _updatePersonalBest.Run(
                 _devicePositions.DevicePointer,
                 _devicePersonalBests.DevicePointer,
                 _deviceGlobalBests.DevicePointer,
                 _particlesCount,
                 _dimensionsCount,
                 Xopt.DevicePointer,
                 Fopt,
                 MinusOnes.DevicePointer,
                 Factor
             );  
        }
    }
}
