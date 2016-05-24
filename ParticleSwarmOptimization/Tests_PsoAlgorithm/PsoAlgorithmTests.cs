using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Algorithm;
using Common;
using ManagedGPU;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tests_Common;

namespace Tests
{
    [TestClass]
    public class PsoAlgorithmTests
    {
        [TestMethod]
        public void RunSimpleAlgorithm()
        {
            var settings = PsoSettingsFactory.QuadraticFunction1DFrom3To5();
            var function = new QuadraticFunction(settings.FunctionParameters);
            var particlesNum = 30;
            var particles = new IParticle[particlesNum];

            for (var i = 0; i < particlesNum; i++)
            {
                particles[i] = ParticleFactory.Create(PsoParticleType.Standard, function.LocationDim,
                    function.FitnessDim);
            }

            var algorithm = new PsoAlgorithm(settings, function, particles.ToArray());
            var result = algorithm.Run();

            Assert.AreEqual(0.0, result.FitnessValue[0], .1);
        }

        [TestMethod]
        public void RunWithGpu()
        {
            var settings = PsoSettingsFactory.QuadraticFunction1DFrom3To5();
            settings.Iterations = 100;
            var function = new QuadraticFunction(settings.FunctionParameters);
            var particlesNum = 300;
            var particles = new IParticle[particlesNum];

            var setup = GpuController.Setup(new CudaParams
            {
                Iterations = 1000,
                LocationDimensions = 1,
                FitnessDimensions = 1,
                ParticlesCount = 300,
                SyncWithCpu = true
            });

            var cudaParticle = setup.Item1;
            var cudaAlgorithm = setup.Item2;

            particles[0] = cudaParticle;
            for (var i = 1; i < particlesNum; i++)
            {
                particles[i] = ParticleFactory.Create(PsoParticleType.Standard, function.LocationDim,
                    function.FitnessDim);
            }

            var algorithm = new PsoAlgorithm(settings, function, particles.ToArray());

            cudaAlgorithm.RunAsync();
            var result = algorithm.Run();
            cudaAlgorithm.Wait();

            Assert.AreEqual(0.0, result.FitnessValue[0], .1);
        }

        [TestMethod]
        public void RunAlgorithmWithTargetValue()
        {
            var settings = PsoSettingsFactory.QuadraticFunction20D();
            settings.IterationsLimitCondition = false;
            settings.TargetValueCondition = true;
            settings.TargetValue = 0.0;
            settings.Epsilon = 0.1;
            var function = new QuadraticFunction(settings.FunctionParameters);
            var particlesNum = 30;
            var particles = new IParticle[particlesNum];
            for (var i = 0; i < particlesNum; i++)
            {
                particles[i] = ParticleFactory.Create(PsoParticleType.Standard, function.LocationDim,
                    function.FitnessDim);
            }
            var algorithm = new PsoAlgorithm(settings, function, particles.ToArray());

            var result = algorithm.Run();

            Assert.AreEqual(0.0, result.FitnessValue[0], .1);
        }

        [TestMethod]
        public void RunAlgorithmWithTargetVAlueAndIterationsLimit()
        {
            var settings = PsoSettingsFactory.QuadraticFunction20D();
            var function = new QuadraticFunction(settings.FunctionParameters);
            RandomGenerator.GetInstance(10);
            settings.Iterations = 10000;
            var particlesNum = 20;
            var particles = new IParticle[particlesNum];
            for (var i = 0; i < particlesNum; i++)
            {
                particles[i] = ParticleFactory.Create(PsoParticleType.Standard, function.LocationDim,
                    function.FitnessDim);
            }
            var algorithm = new PsoAlgorithm(settings, function, particles.ToArray(), new FileLogger("logFile"));

            var result = algorithm.Run();
            Debug.WriteLine(result.FitnessValue[0]);
            foreach (var n in result.Location)
            {
                Debug.Write(n);
            }
            Debug.WriteLine("");
            Assert.AreEqual(0.0, result.FitnessValue[0], .1);

        }
    }
}
