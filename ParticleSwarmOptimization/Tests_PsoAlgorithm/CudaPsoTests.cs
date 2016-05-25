using System;
using System.Linq;
using Algorithm;
using Common;
using ManagedGPU;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tests_Common;

namespace Tests_PsoAlgorithm
{
    [TestClass]
    public class CudaPsoTests
    {
        [TestMethod]
        public void SimpleQuadraticCuda()
        {
            var setup = GpuController.Setup(new CudaParams
            {
                Iterations = 6000,
                LocationDimensions = 1,
                FitnessDimensions = 1,
                ParticlesCount = 1000,
                FitnessFunction = CudaFitnessFunctions.Quadratic,
                SyncWithCpu = false
            });

            var algorithm = setup.Item2;

            var result = algorithm.Run();

            Assert.AreEqual(0.0, result, .01);
        }

        [TestMethod]
        public void TwoDimensionalRosenbrockCuda()
        {
            var setup = GpuController.Setup(new CudaParams
            {
                Iterations = 6000,
                LocationDimensions = 2,
                FitnessDimensions = 1,
                ParticlesCount = 1000,
                FitnessFunction = CudaFitnessFunctions.Rosenbrock,
                SyncWithCpu = false
            });

            var algorithm = setup.Item2;

            var result = algorithm.Run();

            Assert.AreEqual(0.0, result, .01);
        }

        [TestMethod]
        public void MultidimensionalRastriginCuda()
        {
            var setup = GpuController.Setup(new CudaParams
            {
                Iterations = 6000,
                LocationDimensions = 2,
                FitnessDimensions = 1,
                ParticlesCount = 1000,
                FitnessFunction = CudaFitnessFunctions.Rastrigin,
                SyncWithCpu = false
            });

            var algorithm = setup.Item2;

            var result = algorithm.Run();

            Assert.AreEqual(0.0, result, .01);
        }

        [TestMethod]
        public void CpuAndGpuQuadratic()
        {
            var settings = PsoSettingsFactory.QuadraticFunction1DFrom3To5();
            settings.Iterations = 1000;
            var function = new QuadraticFunction(settings.FunctionParameters);
            var particlesNum = 300;
            var particles = new IParticle[particlesNum];

            var setup = GpuController.Setup(new CudaParams
            {
                Iterations = 6000,
                LocationDimensions = 1,
                FitnessDimensions = 1,
                ParticlesCount = 1000,
                FitnessFunction = CudaFitnessFunctions.Quadratic,
                SyncWithCpu = true
            });

            var cudaParticle = setup.Item1;
            var cudaAlgorithm = setup.Item2;

            particles[0] = cudaParticle;
            for (var i = 1; i < particlesNum; i++)
            {
                particles[i] = ParticleFactory.Create(PsoParticleType.Standard, function.LocationDim,
                    function.FitnessDim, function);
            }

            var algorithm = new PsoAlgorithm(settings, function, particles.ToArray());

            cudaAlgorithm.RunAsync();
            var result = algorithm.Run();
            cudaAlgorithm.Wait();

            Assert.AreEqual(0.0, result.FitnessValue[0], .1);
        }
    }
}
