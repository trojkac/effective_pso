using System;
using System.Linq;
using System.Threading;
using Common;
using ManagedCuda;

namespace ManagedGPU
{
    public abstract class GenericCudaAlgorithm : IDisposable
    {
        private int SyncCounter = 200;
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

        protected readonly Random Rng = new Random();

        protected CudaContext Ctx;

        protected abstract void Init();

        protected abstract string KernelFile { get; }

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

        protected void InitContext()
        {
            var size = ParticlesCount * DimensionsCount;

            var threadsNum = 32;
            var blocksNum = ParticlesCount / threadsNum;
            Ctx = new CudaContext(0);

            UpdateVelocity = Ctx.LoadKernel(KernelFile, "updateVelocityKernel");
            UpdateVelocity.GridDimensions = blocksNum;
            UpdateVelocity.BlockDimensions = threadsNum;

            Transpose = Ctx.LoadKernel(KernelFile, "transposeKernel");
            Transpose.GridDimensions = blocksNum;
            Transpose.BlockDimensions = threadsNum;

            HostPositions = new double[size];
            HostVelocities = new double[size];
            HostPersonalBests = new double[size];
            HostPersonalBestValues = new double[ParticlesCount];
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

            for (var i = 0; i < size; i++)
            {
                HostPositions[i] = RandomIn(Rng, -5.0f, 5.0f);
                HostPersonalBests[i] = HostPositions[i];
                HostVelocities[i] = RandomIn(Rng, -2.0f, 2.0f);
            }

            for (var i = 0; i < ParticlesCount; i++)
            {
                HostPersonalBestValues[i] = double.MaxValue;
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

        public void RunAsync()
        {
            if (ThreadHandler != null) return;

            ThreadHandler = new Thread(() =>
            {
                Run();
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

        public double Run()
        {
            InitContext();
            if (SyncWithCpu)
            {
                PushGpuState();
                PullCpuState();
            }

            for (var i = 0; i < Iterations; i++)
            {
                RunUpdateVelocityKernel();
                RunTransposeKernel();

                if (!SyncWithCpu || i % SyncCounter != 0) continue;

                PushGpuState();
                PullCpuState();
            }

            HostPersonalBestValues = DevicePersonalBestValues;
            HostPersonalBests = DevicePersonalBests;

            var bestValue = HostPersonalBestValues.Min();

            return bestValue;
        }

        protected abstract void RunUpdateVelocityKernel();

        protected abstract void RunTransposeKernel();

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
            DevicePositions.Dispose();
            DevicePersonalBestValues.Dispose();
            DeviceVelocities.Dispose();
            DevicePersonalBests.Dispose();
            Ctx.Dispose();
        }

        protected abstract void Cleanup();
    }
}
