using System;
using System.Collections.Generic;
using Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ParticleSwarmOptimizationWrapper;

namespace Tests
{
    [TestClass]
    public class WrapperTests
    {
        [TestMethod]
        public void RunSimpleAlgorithm()
        {
            var function = new QuadraticFunction(new UserFunctionParameters(){Dimension = 5,Coefficients = new []{1.0,1.0,1.0,1.0,1.0}});
            FitnessFunction fitnessFunction = function.Calculate;
            var algorithm = PSOAlgorithm.GetAlgorithm(100, fitnessFunction);
            var particles = new List<Particle>();
            for (int i = 0; i < 20; i++)
            {
                particles.Add(new StandardParticle(2));
            }
            var result = algorithm.Run(particles);

            Assert.AreEqual(0.0, result.FitnessValue, .1);
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
            PSOAlgorithm algorithm = PSOAlgorithm.GetAlgorithm(-1, 0.5, fitnessFunction);
            List<Particle> particles = new List<Particle>();
            for (int i = 0; i < 20; i++)
            {
                particles.Add(new StandardParticle(2));
            }
            var result = algorithm.Run(particles);

            Assert.AreEqual(-.5, result.FitnessValue, .5);
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
            PSOAlgorithm algorithm = PSOAlgorithm.GetAlgorithm(100, -0.5, 0.1, fitnessFunction);
            List<Particle> particles = new List<Particle>();
            for (int i = 0; i < 20; i++)
            {
                particles.Add(new StandardParticle(2));
            }
            var result = algorithm.Run(particles);

            Assert.AreEqual(-.5, result.FitnessValue, .5);
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
            PSOAlgorithm algorithm = PSOAlgorithm.GetAlgorithm(4000, fitnessFunction);
            List<Particle> particles = new List<Particle>();
            for (int i = 0; i < 20; i++)
            {
                particles.Add(new StandardParticle(2));
            }
            var result = algorithm.Run(particles);

            Assert.AreEqual(-.50, result.FitnessValue, .5);
        }
    }
}
