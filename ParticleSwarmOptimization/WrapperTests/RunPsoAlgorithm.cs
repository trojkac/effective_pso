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
            FitnessFunction fitnessFunction = values => Math.Sin(values[0] * values[0] + values[1] * values[1]);
            PSOAlgorithm algorithm = PSOAlgorithm.GetAlgorithm(100,fitnessFunction);
            List<Particle> particles = new List<Particle>();
            for (int i = 0; i < 20; i++)
            {
                particles.Add(new FullyInformedParticle(2));
            }
            var result = algorithm.Run(particles);

            Assert.AreEqual(1.0, result.Item2,.1);
        }
    }
}
