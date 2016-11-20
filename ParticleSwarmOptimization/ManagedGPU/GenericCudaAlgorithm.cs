using System;
using System.Linq;
using System.Threading;
using Common;
using ManagedCuda;
using ManagedCuda.BasicTypes;


namespace ManagedGPU
{
    public abstract class GenericCudaAlgorithm : IDisposable
    {
        private int SyncCounter = 100;
        protected bool SyncWithCpu;

        protected IFitnessFunction<double[], double[]> FitnessFunction;

        protected Thread ThreadHandler;

        protected StateProxy Proxy;

        protected int FunctionNumber;

        protected int InstanceNumber;

        protected int ParticlesCount;

        protected int DimensionsCount;

        protected int Iterations;

        protected CudaKernel UpdateVelocity;

        protected CudaKernel Transpose;

        protected double[] HostPositions;

        protected double[] HostVelocities;

        protected double[] HostPersonalBests;

        protected double[] HostPersonalBestValues;

        protected int[] HostNeighbors;

        protected CudaDeviceVariable<double> DevicePositions;

        protected CudaDeviceVariable<double> DeviceVelocities;

        protected CudaDeviceVariable<double> DevicePersonalBests;

        protected CudaDeviceVariable<double> DevicePersonalBestValues;

        protected CudaDeviceVariable<int> DeviceNeighbors; 

        protected readonly RandomGenerator Random = RandomGenerator.GetInstance();

        protected CudaContext Ctx;

        protected abstract void Init();

        protected abstract string KernelFile { get; }

        protected CudaDeviceVariable<double> Xopt;
        protected double Fopt;

        private CudaDeviceVariable<double> _phis1;
        private CudaDeviceVariable<double> _phis2;



        protected GenericCudaAlgorithm(CudaParams parameters, StateProxy proxy)
        {
            Proxy = proxy;
            ParticlesCount = parameters.ParticlesCount;
            DimensionsCount = parameters.LocationDimensions;
            Iterations = parameters.Iterations;
            FitnessFunction = parameters.FitnessFunction;
            SyncWithCpu = parameters.SyncWithCpu;
            FunctionNumber = parameters.FunctionNumber;
            InstanceNumber = parameters.InstanceNumber;

        }

        protected void InitializePhis()
        {
            _phis1 = new CudaDeviceVariable<double>(ParticlesCount * DimensionsCount);
            _phis2 = new CudaDeviceVariable<double>(ParticlesCount * DimensionsCount);
        }

        protected void InitContext()
        {
            var size = ParticlesCount * DimensionsCount;

            var threadsNum = 32;
            var blocksNum = ParticlesCount / threadsNum;
            Ctx = new CudaContext(0);

            UpdateVelocity = Ctx.LoadKernel("update_velocity_kernel.ptx", "updateVelocityKernel");
            UpdateVelocity.GridDimensions = blocksNum;
            UpdateVelocity.BlockDimensions = threadsNum;

            Transpose = Ctx.LoadKernel(KernelFile, "transposeKernel");
            Transpose.GridDimensions = blocksNum;
            Transpose.BlockDimensions = threadsNum;

            HostPositions = Random.RandomVector(size, -5.0, 5.0);
            HostVelocities = Random.RandomVector(size, -2.0, 2.0);
            HostPersonalBests = (double[]) HostPositions.Clone();
            HostPersonalBestValues = Enumerable.Repeat(double.MaxValue,ParticlesCount).ToArray();

            HostNeighbors = new int[ParticlesCount * 2];

            for (var i = 0; i < ParticlesCount*2; i += 2)
            {
                int left, right;

                if (i == 0)
                    left = ParticlesCount - 1;
                else
                    left = i - 1;

                if (i == ParticlesCount - 1)
                    right = 0;
                else
                    right = i + 1;

                HostNeighbors[i] = left;
                HostNeighbors[i + 1] = right;
            }

            DevicePositions = HostPositions;
            DeviceVelocities = HostVelocities;
            DevicePersonalBests = HostPersonalBests;
            DevicePersonalBestValues = HostPersonalBestValues;
            DeviceNeighbors = HostNeighbors;

            Init();
        }

        protected void PullCpuState()
        {
            for (var i = 0; i < DimensionsCount; i++)
                DevicePositions[i] = Proxy.CpuState.Location[i];
        }

        protected void PushGpuState()
        {
            var bestVal = DevicePersonalBestValues[0];
            var bestLoc = new double[DimensionsCount];

            for (int i = 0; i < DimensionsCount; i++)
            {
                bestLoc[i] = DevicePersonalBests[i];
            }

            Proxy.GpuState = new ParticleState(bestLoc, new []{ bestVal }); 
        }

        public void RunAsync(CancellationToken token = new CancellationToken())
        {
            if (ThreadHandler != null) return;

            ThreadHandler = new Thread(() =>
            {
                Run(token);
                Cleanup();
            });

            ThreadHandler.Start();
        }

        public void Wait()
        {
            if (ThreadHandler != null)
                ThreadHandler.Join();
        }

        public void RunAsyncAndWait()
        {
            RunAsync();
            Wait();
        }

        public void Abort()
        {
            if (ThreadHandler != null)
                ThreadHandler.Abort();
        }

        public double Run(CancellationToken token = new CancellationToken())
        {
            InitContext();
            InitializePhis();

            if (SyncWithCpu)
            {
                PushGpuState();
                PullCpuState();
            }
            var i = 0;
            try
            {

                for (i = 0; i < Iterations; i++)
                {
                    token.ThrowIfCancellationRequested();
                    RunUpdateVelocityKernel();
                    RunTransposeKernel();

                    if (!SyncWithCpu || i % SyncCounter != 0) continue;

                    PushGpuState();
                    PullCpuState();
                }
            }
            catch (OperationCanceledException ex)
            {

            }
            Console.WriteLine("Performed CUDA Iterations: {0}", i);
            HostPersonalBestValues = DevicePersonalBestValues;
            HostPersonalBests = DevicePersonalBests;

            var bestValue = HostPersonalBestValues.Min();

            return bestValue;
        }

        protected  void RunUpdateVelocityKernel()
        {
            assignRandoms();
            UpdateVelocity.Run(
                    DevicePositions.DevicePointer,
                    DeviceVelocities.DevicePointer,
                    DevicePersonalBests.DevicePointer,
                    DevicePersonalBestValues.DevicePointer,
                    DeviceNeighbors.DevicePointer,
                    ParticlesCount,
                    DimensionsCount,
                    _phis1.DevicePointer,
                    _phis2.DevicePointer
                );
        }
        private void assignRandoms()
        {
            var hostPhis1 = RandomGenerator.GetInstance().RandomVector(DimensionsCount * ParticlesCount, 0, 1);
            var hostPhis2 = RandomGenerator.GetInstance().RandomVector(DimensionsCount * ParticlesCount, 0, 1);
            _phis1.CopyToDevice(hostPhis1);
            _phis2.CopyToDevice(hostPhis2);

        }

        protected abstract void RunTransposeKernel();

        public virtual void Dispose()
        {
            DevicePositions.Dispose();
            DevicePersonalBestValues.Dispose();
            DeviceVelocities.Dispose();
            DevicePersonalBests.Dispose();
            _phis1.Dispose();
            _phis2.Dispose();
            Ctx.Dispose();
        }

        protected abstract void Cleanup();

    }
}
