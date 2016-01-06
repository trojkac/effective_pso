using System;
using System.Linq;
using Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using PsoService;

namespace Tests
{
    /// <summary>
    /// Summary description for ProxyParticleTests
    /// </summary>
    [TestClass]
    public class ProxyParticleTests
    {
        public ProxyParticleTests()
        {
            //
            // TODO: Add constructor logic here
            //
        }

        private TestContext testContextInstance;

        /// <summary>
        ///Gets or sets the test context which provides
        ///information about and functionality for the current test run.
        ///</summary>
        public TestContext TestContext
        {
            get
            {
                return testContextInstance;
            }
            set
            {
                testContextInstance = value;
            }
        }

        #region Additional test attributes
        //
        // You can use the following additional attributes as you write your tests:
        //
        // Use ClassInitialize to run code before running the first test in the class
        // [ClassInitialize()]
        // public static void MyClassInitialize(TestContext testContext) { }
        //
        // Use ClassCleanup to run code after all tests in a class have run
        // [ClassCleanup()]
        // public static void MyClassCleanup() { }
        //
        // Use TestInitialize to run code before running each test 
        // [TestInitialize()]
        // public void MyTestInitialize() { }
        //
        // Use TestCleanup to run code after each test has run
        // [TestCleanup()]
        // public void MyTestCleanup() { }
        //
        #endregion

        [TestMethod]
        public void BasicCommunication()
        {
            var particle1 = ProxyParticle.CreateProxyParticle(1);
            var particle2 = ProxyParticle.CreateProxyParticle(2);
            particle1.UpdateRemoteAddress(particle2.Address);
            var state = new ParticleState {FitnessValue = 0.0, Location = new[] {0.0, 0.4}};
            particle2.UpdateBestState(state);

            particle2.Open();
            particle1.GetRemoteBest();
            particle2.Close();

            Assert.AreEqual(state.FitnessValue,particle1.GetBestState().FitnessValue);
            Assert.AreEqual(true,particle1.GetBestState().Location.SequenceEqual(state.Location));

        }

        [TestMethod]
        public void CommunicationInAlgorithm()
        {
            var particle1 = ProxyParticle.CreateProxyParticle(1);
            var particle2 = ProxyParticle.CreateProxyParticle(2);
            particle1.UpdateRemoteAddress(particle2.Address);
            var state = new ParticleState {FitnessValue = 5.0, Location = new[] {0.0}};
            particle2.UpdateBestState(state);

            particle2.Open();

            var controller = new Controller.Controller();
            
            var result = controller.Run(x => -x[0]*x[0]+5,
                new PsoSettings()
                {
                    Dimensions = 1,
                    Epsilon = 0,
                    Iterations = 2,
                    IterationsLimitCondition = true,
                    Particles = new[] {new Tuple<PsoParticleType, int>(PsoParticleType.Standard, 1)},
                }, new []{particle1}
                );

            Assert.AreEqual(state.FitnessValue, result.FitnessValue);
            Assert.AreEqual(true, result.Location.SequenceEqual(state.Location));

        }
    }
}
