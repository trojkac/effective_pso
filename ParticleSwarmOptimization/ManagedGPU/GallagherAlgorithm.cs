using ManagedCuda;

namespace ManagedGPU
{
    class GallagherAlgorithm : GenericCudaAlgorithm
    {
        public double PeaksCount { get; set; }

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

        public GallagherAlgorithm(CudaParams parameters, StateProxy proxy)
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

            Rotation = new CudaDeviceVariable<double>(_dimensionsCount * _dimensionsCount);
            PeakValues = new CudaDeviceVariable<double>((int)PeaksCount);
            XLocal = new CudaDeviceVariable<double>(_dimensionsCount * (int)PeaksCount);
            ArrScales = new CudaDeviceVariable<double>(_dimensionsCount * (int)PeaksCount);

            long rseed = _functionNumber + 10000 * _instanceNumber;

            initKernel.Run(
                _dimensionsCount, 
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
                Rotation.DevicePointer,
                PeaksCount,
                PeakValues.DevicePointer,
                XLocal.DevicePointer,
                ArrScales.DevicePointer
             );  
        }
    }
}
