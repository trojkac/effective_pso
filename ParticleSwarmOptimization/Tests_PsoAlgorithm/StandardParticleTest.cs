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
            var particle = new StandardParticle();
            var initState = ParticleStateFactory.Create(4, 1);
            initState.Location = new[] {0,1,2,3.0};
            particle.Init(initState, new[] {-1.0, 2, -3, 4});

            particle.Translate();
            var result = new[] {-1, 3, -1, 7.0};
            for (int i = 0; i < 4; i++)
            {
                Assert.AreEqual(particle.CurrentState.Location[i], result[i]);

            }

        }
    }
}
