using System;
using ManagedCuda;

namespace ManagedGPU
{
    class WeierstrassAlgorithm : GenericCudaAlgorithm
    {
        protected CudaDeviceVariable<double> Xopt;
        protected CudaDeviceVariable<double> M;
        protected CudaDeviceVariable<double> B;

        protected double Fopt;

        protected double Penalty = 10.0;

        protected double F0 = 0.0;
        protected CudaDeviceVariable<double> Ak;
        protected CudaDeviceVariable<double> Bk;

        public override void Dispose()
        {
            Xopt.Dispose();
            M.Dispose();
            B.Dispose();
            Ak.Dispose();
            Bk.Dispose();
            base.Dispose();
        }

        public WeierstrassAlgorithm(CudaParams parameters, StateProxy proxy)
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

            double[] h_ak = new double[12];
            double[] h_bk = new double[12];

            for (int i = 0; i < 12; i++)
            {
                h_ak[i] = Math.Pow(0.5, i);
                h_bk[i] = Math.Pow(3.0, i);
                F0 += h_ak[i]*Math.Cos(Math.PI*h_bk[i]);
            }

            Ak = h_ak;
            Bk = h_bk;

            initKernel.Run(
                _dimensionsCount, 
                rseed, 
                _functionNumber, 
                _instanceNumber,
                M.DevicePointer,
                B.DevicePointer,
                Xopt.DevicePointer,
                d_fopt.DevicePointer);

            double[] fopt_arr = d_fopt;

            Fopt = fopt_arr[0];
        }

        protected override string KernelFile
        {
            get { return "f16_weierstrass_kernel.ptx"; }
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
                 Xopt,
                 Fopt,
                 Penalty,
                 M,
                 B,
                 Ak,
                 Bk,
                 F0
             );  
        }
    }
}
