using System;
using System.Linq;
using System.Threading;
using Common;
using ManagedCuda;

namespace ManagedGPU
{
    public abstract class GenericCudaAlgorithm : IDisposable
    {
        protected bool _syncWithCpu;

        protected IFitnessFunction<double[], double[]> _fitnessFunction;

        protected Thread _threadHandler;

        protected StateProxy _proxy;

        protected int _functionNumber;

        protected int _instanceNumber;

        protected int _particlesCount;

        protected int _dimensionsCount;

        protected int _iterations;

        protected CudaKernel _updateParticle;

        protected CudaKernel _updatePersonalBest;

        protected double[] _hostPositions;

        protected double[] _hostVelocities;

        protected double[] _hostPersonalBests;

        protected double[] _hostGlobalBests;

        protected CudaDeviceVariable<double> _devicePositions;

        protected CudaDeviceVariable<double> _deviceVelocities;

        protected CudaDeviceVariable<double> _devicePersonalBests;

        protected CudaDeviceVariable<double> _deviceGlobalBests;

        protected readonly Random _rng = new Random();

        protected CudaContext ctx;

        protected abstract void Init();

        protected abstract string KernelFile { get; }

        protected void InitContext()
        {
            var size = _particlesCount * _dimensionsCount;

            var threadsNum = 32;
            var blocksNum = _particlesCount / threadsNum;
            ctx = new CudaContext(0);

            _updateParticle = ctx.LoadKernel(KernelFile, "kernelUpdateParticle");
            _updateParticle.GridDimensions = blocksNum;
            _updateParticle.BlockDimensions = threadsNum;

            _updatePersonalBest = ctx.LoadKernel(KernelFile, "kernelUpdatePBest");
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
                _hostVelocities[i] = RandomIn(_rng, -2.0f, 2.0f);
            }

            for (var i = 0; i < _dimensionsCount; i++)
                _hostGlobalBests[i] = _hostPersonalBests[i];

            _devicePositions = _hostPositions;
            _deviceVelocities = _hostVelocities;
            _devicePersonalBests = _hostPersonalBests;
            _deviceGlobalBests = _hostGlobalBests;

            Init();
        }

        protected void PullCpuState()
        {
            _hostPositions = _devicePositions;

            for (var i = 0; i < _dimensionsCount; i++)
                _hostPositions[i] = _proxy.CpuState.Location[i];

            _devicePositions = _hostPositions;
        }

        protected void PushGpuState()
        {
            _hostPersonalBests = _devicePersonalBests;
            var location = _hostPersonalBests.Take(_dimensionsCount).ToArray();
            var fitnessValue = _fitnessFunction.Evaluate(location);
            _proxy.GpuState = new ParticleState(location, fitnessValue); 
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

            return _fitnessFunction.Evaluate(_hostGlobalBests)[0];
        }

        protected void Step()
        {
            var size = _particlesCount*_dimensionsCount;
            var temp = new double[_dimensionsCount];

            RunUpdateParticleKernel();

            RunUpdatePersonalBestKernel();

            _hostPersonalBests = _devicePersonalBests;

            for (var i = 0; i < size; i += _dimensionsCount)
            {
                for (var k = 0; k < _dimensionsCount; k++)
                    temp[k] = _hostPersonalBests[i + k];

                if (_fitnessFunction.Evaluate(GetClampedLocation(temp))[0] < _fitnessFunction.Evaluate(_hostGlobalBests)[0])
                {
                    for (var k = 0; k < _dimensionsCount; k++)
                        _hostGlobalBests[k] = GetClampedLocation(temp)[k];
                }
            }

            _deviceGlobalBests = _hostGlobalBests;
        }

        private double[] GetClampedLocation(double[] vector)
        {
            if (vector == null) return vector;
            return vector.Select((x, i) => Math.Min(Math.Max(x, -5.0), 5.0)).ToArray();
        }

        protected abstract void RunUpdateParticleKernel();

        protected abstract void RunUpdatePersonalBestKernel();

        protected static double RandomIn(Random rng, double min, double max)
        {
            return min + Random(rng) * (max - min);
        }

        protected static double Random(Random rng)
        {
            return rng.NextDouble();
        }

        public virtual void Dispose()
        {
            _devicePositions.Dispose();
            _deviceGlobalBests.Dispose();
            _deviceVelocities.Dispose();
            _devicePersonalBests.Dispose();
            ctx.Dispose();
        }
    }
}
