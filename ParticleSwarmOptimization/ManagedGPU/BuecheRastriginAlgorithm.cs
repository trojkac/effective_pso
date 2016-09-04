using ManagedCuda;

namespace ManagedGPU
{
    class BuecheRastriginAlgorithm : GenericCudaAlgorithm
    {
        protected CudaDeviceVariable<double> Xopt;

        protected double Fopt;
        protected double Penalty = 100.0;

        public override void Dispose()
        {
            Xopt.Dispose();
            base.Dispose();
        }

        public BuecheRastriginAlgorithm(CudaParams parameters, StateProxy proxy)
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

            long rseed = (long)(3 + 10000 * _instanceNumber);

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
            get { return "f4_buche_rastrigin_kernel.ptx"; }
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
                 Penalty
             );  
        }
    }
}
