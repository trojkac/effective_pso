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

        private double rseed;

        public SchwefelAlgorithm(CudaParams parameters, StateProxy proxy)
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
            B = new CudaDeviceVariable<double>(_dimensionsCount);
            M = new CudaDeviceVariable<double>(_dimensionsCount * _dimensionsCount);
            Xopt = new CudaDeviceVariable<double>(_dimensionsCount);
            Tmp1 = new CudaDeviceVariable<double>(_dimensionsCount);
            Tmp2 = new CudaDeviceVariable<double>(_dimensionsCount);

            var d_fopt = new CudaDeviceVariable<double>(1);

            rseed = _functionNumber + 10000 * _instanceNumber;

            initKernel.Run(
                _dimensionsCount, 
                rseed, 
                _functionNumber, 
                _instanceNumber, 
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
