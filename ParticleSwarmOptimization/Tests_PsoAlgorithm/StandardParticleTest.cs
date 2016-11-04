using System;
using Algorithm;
using Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Tests_PsoAlgorithm
{
    [TestClass]
    public class StandardParticleTest
    {
        [TestMethod]
        public void TransposeProperly()
        {
            var rand = RandomGenerator.GetInstance();
            FitnessFunctionEvaluation ffe = values => values;
            var function = new FitnessFunction(ffe);
            var initVelocity = new[] {-1.0, 2, -3, 4};
            var particle = ParticleFactory.Create(PsoParticleType.Standard, 4, 1, function,1e-10,100, null, initVelocity);
            var initState = particle.CurrentState;
            particle.Transpose(function);
            for (int i = 0; i < 4; i++)
            {
                Assert.AreEqual(particle.CurrentState.Location[i], initVelocity[i]+initState.Location[i]);

            }

        }
    }
}
