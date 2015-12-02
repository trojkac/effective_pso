using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ParticleSwarmOptimizationWrapper;
namespace WrapperTests
{
    [TestClass]
    public class RunPsoAlgorithm
    {
        [TestMethod]
        public void RunSimpleAlgorithm()
        {
            FitnessFunction fitnessFunction = values =>
            {
                var x = values[0] * values[0];
                var y = values[1] * values[1];
                return Math.Sin(x + y) / (x * y + 1);
            };
            PSOAlgorithm algorithm = PSOAlgorithm.GetAlgorithm(100, fitnessFunction);
            List<Particle> particles = new List<Particle>();
            for (int i = 0; i < 20; i++)
            {
                particles.Add(new FullyInformedParticle(2));
            }
            var result = algorithm.Run(particles);

            Assert.AreEqual(1.0, result.Item2, .1);
        }
        [TestMethod]
        public void RunAlgorithmWithTargetValue()
        {
            FitnessFunction fitnessFunction = values =>
            {
                var x = values[0] * values[0];
                var y = values[1] * values[1];
                return Math.Sin(x + y) / (x * y + 1);
            };
            PSOAlgorithm algorithm = PSOAlgorithm.GetAlgorithm(1.0, 0.1, fitnessFunction);
            List<Particle> particles = new List<Particle>();
            for (int i = 0; i < 20; i++)
            {
                particles.Add(new FullyInformedParticle(2));
            }
            var result = algorithm.Run(particles);

            Assert.AreEqual(1.0, result.Item2, .1);
        }

        [TestMethod]
        public void RunAlgorithmWithTargetVAlueAndIterationsLimit()
        {
            FitnessFunction fitnessFunction = values =>
            {
                var x = values[0] * values[0];
                var y = values[1] * values[1];
                return Math.Sin(x + y) / (x * y + 1);
            };
            PSOAlgorithm algorithm = PSOAlgorithm.GetAlgorithm(100, 1.0, 0.1 ,fitnessFunction);
            List<Particle> particles = new List<Particle>();
            for (int i = 0; i < 20; i++)
            {
                particles.Add(new FullyInformedParticle(2));
            }
            var result = algorithm.Run(particles);

            Assert.AreEqual(1.0, result.Item2, .1);
        }

        [TestMethod]
        public void RunSimpleAlgorithmWith1000Iterations()
        {
            FitnessFunction fitnessFunction = values =>
            {
                var x = values[0] * values[0];
                var y = values[1] * values[1];
                return Math.Sin(x + y) / (x * y + 1);
            };
            PSOAlgorithm algorithm = PSOAlgorithm.GetAlgorithm(1000, fitnessFunction);
            List<Particle> particles = new List<Particle>();
            for (int i = 0; i < 20; i++)
            {
                particles.Add(new FullyInformedParticle(2));
            }
            var result = algorithm.Run(particles);

            Assert.AreEqual(1.0, result.Item2, .1);
        }
    }
}
