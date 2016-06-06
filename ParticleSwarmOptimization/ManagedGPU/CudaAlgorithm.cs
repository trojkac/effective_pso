using System;
using System.Linq;
using System.Threading;
using Common;
using ManagedCuda;

namespace ManagedGPU
{
    public class CudaAlgorithm
    {
        private readonly bool _syncWithCpu;

        private readonly ICudaFitnessFunction _fitnessFunction;

        private Thread _threadHandler;

        private readonly StateProxy _proxy;

        private readonly int _particlesCount;

        private readonly int _dimensionsCount;

        private readonly int _iterations;

        private CudaKernel _updateParticle;

        private CudaKernel _updatePersonalBest;

        private double[] _hostPositions;

        private double[] _hostVelocities;

        private double[] _hostPersonalBests;

        private double[] _hostGlobalBests;

        private CudaDeviceVariable<double> _devicePositions;

        private CudaDeviceVariable<double> _deviceVelocities;

        private CudaDeviceVariable<double> _devicePersonalBests;

        private CudaDeviceVariable<double> _deviceGlobalBests;

        private readonly Random _rng = new Random();

        internal CudaAlgorithm(CudaParams parameters, StateProxy proxy)
        {
            _proxy = proxy;
            _particlesCount = parameters.ParticlesCount;
            _dimensionsCount = parameters.LocationDimensions;
            _iterations = parameters.Iterations;
            _fitnessFunction = parameters.FitnessFunction;
            _syncWithCpu = parameters.SyncWithCpu;
        }

        private void InitContext()
        {
            var size = _particlesCount * _dimensionsCount;

            var ctx = new CudaContext(0);

            var deviceInfo = ctx.GetDeviceInfo();

            int threadsNum = (int)deviceInfo.MaxBlockDim.x;
            var blocksNum = (int)Math.Ceiling((double)_particlesCount / threadsNum);

            _updateParticle = ctx.LoadKernel(_fitnessFunction.KernelFile, "kernelUpdateParticle");
            _updateParticle.GridDimensions = blocksNum;
            _updateParticle.BlockDimensions = threadsNum;

            _updatePersonalBest = ctx.LoadKernel(_fitnessFunction.KernelFile, "kernelUpdatePBest");
            _updatePersonalBest.GridDimensions = blocksNum;
            _updatePersonalBest.BlockDimensions = threadsNum;

            _hostPositions = new double[size];
            _hostVelocities = new double[size];
            _hostPersonalBests = new double[size];
            _hostGlobalBests = new double[_dimensionsCount];

            for (var i = 0; i < size; i++)
            {
                _hostPositions[i] = RandomIn(_rng, -5.0f, 5.0f);
                _hostPersonalBests[i] = _hostPositions[i];
                _hostVelocities[i] = 0.0f;
            }

            for (var i = 0; i < _dimensionsCount; i++)
                _hostGlobalBests[i] = _hostPersonalBests[i];

            _devicePositions = _hostPositions;
            _deviceVelocities = _hostVelocities;
            _devicePersonalBests = _hostPersonalBests;
            _deviceGlobalBests = _hostGlobalBests;
        }

        private void PullCpuState()
        {
            _hostPositions = _devicePositions;

            for (var i = 0; i < _dimensionsCount; i++)
                _hostPositions[i] = _proxy.CpuState.Location[i];

            _devicePositions = _hostPositions;
        }

        private void PushGpuState()
        {
            _hostPositions = _devicePositions;
            var temp = _hostPositions.Take(_dimensionsCount).ToArray();
            var location = _hostPositions.Take(_dimensionsCount).ToArray();
            var fitnessValue = new[] {_fitnessFunction.HostFitnessFunction(location)};
            _proxy.GpuState = new ParticleState(location, fitnessValue); ;
        }

        public void RunAsync()
        {
            if (_threadHandler != null) return;

            _threadHandler = new Thread(() =>
            {
                InitContext();
                Run();
            });

            _threadHandler.Start();
        }

        public void Wait()
        {
            if (_threadHandler != null)
                _threadHandler.Join();
        }

        public void RunAsyncAndWait()
        {
            RunAsync();
            Wait();
        }

        public void Abort()
        {
            if (_threadHandler != null)
                _threadHandler.Abort();
        }

        public double Run()
        {
            InitContext();

            if (_syncWithCpu)
            {
                PushGpuState();
                PullCpuState();
            }

            for (var i = 0; i < _iterations; i++)
            {
                Step();

                if (!_syncWithCpu) continue;

                PushGpuState();
                PullCpuState();
            }

            _hostGlobalBests = _deviceGlobalBests;

            return _fitnessFunction.HostFitnessFunction(_hostGlobalBests);
        }

        private void Step()
        {
            var size = _particlesCount*_dimensionsCount;
            var temp = new double[_dimensionsCount];

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

            _updatePersonalBest.Run(
                _devicePositions.DevicePointer,
                _devicePersonalBests.DevicePointer,
                _deviceGlobalBests.DevicePointer,
                _particlesCount,
                _dimensionsCount
            );

            _hostPersonalBests = _devicePersonalBests;

            for (var i = 0; i < size; i += _dimensionsCount)
            {
                for (var k = 0; k < _dimensionsCount; k++)
                    temp[k] = _hostPersonalBests[i + k];

                if (_fitnessFunction.HostFitnessFunction(temp) < _fitnessFunction.HostFitnessFunction(_hostGlobalBests))
                {
                    for (var k = 0; k < _dimensionsCount; k++)
                        _hostGlobalBests[k] = temp[k];
                }
            }

            _deviceGlobalBests = _hostGlobalBests;
        }

        private static double RandomIn(Random rng, double min, double max)
        {
            return min + Random(rng) * (max - min);
        }

        private static double Random(Random rng)
        {
            return rng.NextDouble();
        }
    }
}
