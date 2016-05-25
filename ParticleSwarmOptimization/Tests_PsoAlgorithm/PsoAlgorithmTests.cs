using System.Diagnostics;
using System.Linq;
using Algorithm;
using Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tests_Common;

namespace Tests_PsoAlgorithm
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
                    function.FitnessDim,function);
            }

            var algorithm = new PsoAlgorithm(settings, function, particles.ToArray());
            var result = algorithm.Run();

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
                    function.FitnessDim,function);
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
            settings.Iterations = 1000;
            var particlesNum = 20;
            var particles = new IParticle[particlesNum];
            for (var i = 0; i < particlesNum; i++)
            {
                particles[i] = ParticleFactory.Create(PsoParticleType.Standard, function.LocationDim,
                    function.FitnessDim,function);
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
