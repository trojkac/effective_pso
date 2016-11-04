using System;
using System.Linq;
using Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using PsoService;
using Tests_Common;

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
            particle1.UpdateRemoteAddress(new Uri(String.Format("net.tcp://127.0.0.1:{0}{1}", particle2.Address.Port, particle2.Address.LocalPath)));
            var state = new ParticleState {FitnessValue = new []{ 0.0 }, Location = new[] {0.0, 0.4}};
            particle2.UpdateBestState(state);

            particle2.Open();
            particle1.GetRemoteBest();
            particle2.Close();

            Assert.AreEqual(state.FitnessValue[0],particle1.GetBestState().FitnessValue[0]);
            Assert.AreEqual(true,particle1.GetBestState().Location.SequenceEqual(state.Location));

        }

        [TestMethod]
        public void CommunicationInAlgorithm()
        {
            var particle1 = ProxyParticle.CreateProxyParticle(1);
            var particle2 = ProxyParticle.CreateProxyParticle(2);
            particle1.UpdateRemoteAddress(new Uri(String.Format("net.tcp://127.0.0.1:{0}{1}", particle2.Address.Port, particle2.Address.LocalPath)));
            var state = new ParticleState {FitnessValue = new []{ 9.0 }, Location = new[] {3.0}};
            particle2.UpdateBestState(state);

            particle2.Open();

            var controller = new Controller.PsoController(123);
            var settings = PsoSettingsFactory.QuadraticFunction1DFrom3To5();
            settings.Iterations = 1;
            controller.Run(settings, new []{particle1});
            
            var result = controller.RunningAlgorithm.Result;
            Assert.AreEqual(state.FitnessValue[0], result.FitnessValue[0]);
            Assert.AreEqual(true, result.Location.SequenceEqual(state.Location));

        }
    }
}
