﻿using ManagedCuda;

namespace ManagedGPU
{
    class StepEllipsoidAlgorithm : GenericCudaAlgorithm
    {
        protected CudaDeviceVariable<double> Rotation1;
        protected CudaDeviceVariable<double> Rotation2;
        protected CudaDeviceVariable<double> Xopt;

        protected double Fopt;

        public override void Dispose()
        {
            Rotation1.Dispose();
            Rotation2.Dispose();
            Xopt.Dispose();
            base.Dispose();
        }

        public StepEllipsoidAlgorithm(CudaParams parameters, StateProxy proxy)
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
            Rotation2 = new CudaDeviceVariable<double>(_dimensionsCount * _dimensionsCount);
            Rotation1 = new CudaDeviceVariable<double>(_dimensionsCount * _dimensionsCount);
            Xopt = new CudaDeviceVariable<double>(_dimensionsCount);

            var d_fopt = new CudaDeviceVariable<double>(1);

            long rseed = _functionNumber + 10000 * _instanceNumber;

            initKernel.Run(
                _dimensionsCount, 
                rseed, 
                _functionNumber, 
                _instanceNumber, 
                Rotation1.DevicePointer, 
                Rotation2.DevicePointer, 
                Xopt.DevicePointer,
                d_fopt.DevicePointer);

            double[] fopt_arr = d_fopt;

            Fopt = fopt_arr[0];
        }

        protected override string KernelFile
        {
            get { return "f7_step_ellipsoid.ptx"; }
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
                 Rotation1.DevicePointer,
                 Rotation2.DevicePointer,
                 Fopt
             );  
        }
    }
}
