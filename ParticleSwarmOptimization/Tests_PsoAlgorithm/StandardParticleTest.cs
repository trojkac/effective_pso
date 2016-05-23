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
            var particle = ParticleFactory.Create(PsoParticleType.Standard, 4,1, new FitnessFunction(ffe));
            var initState = new ParticleState(rand.RandomVector(4),rand.RandomVector(1));
            particle.Translate();
            var result = new[] {-1, 3, -1, 7.0};
            for (int i = 0; i < 4; i++)
            {
                Assert.AreEqual(particle.CurrentState.Location[i], result[i]);

            }

        }
    }
}
