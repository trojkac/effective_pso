using System;
using ManagedCuda;

namespace ManagedGPU
{
    public class CudaAlgorithm
    {
        private int _particlesCount;

        private int _dimensionsCount;

        private readonly CudaKernel _updateParticle;

        private readonly CudaKernel _updatePersonalBest;

        private float[] _hostPositions;

        private float[] _hostVelocities;

        private float[] _hostPersonalBests;

        private float[] _hostGlobalBests;

        private CudaDeviceVariable<float> _devicePositions;

        private CudaDeviceVariable<float> _deviceVelocities;

        private CudaDeviceVariable<float> _devicePersonalBests;

        private CudaDeviceVariable<float> _deviceGlobalBests;

        private readonly Random _rng = new Random();

        public CudaAlgorithm()
        {
            var size = _particlesCount * _dimensionsCount;

            var threadsNum = 32;
            var blocksNum = _particlesCount / threadsNum;

            var ctx = new CudaContext(0);

            _updateParticle = ctx.LoadKernel("psoKernel.ptx", "kernelUpdateParticle");
            _updateParticle.GridDimensions = blocksNum;
            _updateParticle.BlockDimensions = threadsNum;

            _updatePersonalBest = ctx.LoadKernel("psoKernel.ptx", "kernelUpdatePBest");
            _updatePersonalBest.GridDimensions = blocksNum;
            _updatePersonalBest.BlockDimensions = threadsNum;

            _hostPositions = new float[size];
            _hostVelocities = new float[size];
            _hostPersonalBests = new float[size];
            _hostGlobalBests = new float[_dimensionsCount];

            for (var i = 0; i < size; i++)
            {
                _hostPositions[i] = RandomIn(_rng, 3.0f, 5.0f);
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

        public void Step()
        {
            var size = _particlesCount*_dimensionsCount;
            var temp = new float[_dimensionsCount];

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

                if (HostFitnessFunction(temp, _dimensionsCount) < HostFitnessFunction(_hostGlobalBests, _dimensionsCount))
                {
                    for (var k = 0; k < _dimensionsCount; k++)
                        _hostGlobalBests[k] = temp[k];
                }
            }

            _deviceGlobalBests = _hostGlobalBests;
        }

        private static float Random(Random rng)
        {
            return (float)rng.NextDouble();
        }

        private static float RandomIn(Random rng, float min, float max)
        {
            return min + Random(rng) * (max - min);
        }

        private static float HostFitnessFunction(float[] x, int dimensionCount)
        {
            var res = 0.0f;

            for (var i = 0; i < dimensionCount; i++)
                res += x[i] * x[i];

            return res;
        }
    }
}
