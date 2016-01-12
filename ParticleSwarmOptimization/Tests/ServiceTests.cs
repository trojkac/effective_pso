using System;
using System.Net;
using Controller;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Node;


namespace Tests
{
    [TestClass]
    public class ServiceTests
    {
        [TestMethod]
        public void ClusterRegister()
        {
            VCpuManager vcpu1 = new VCpuManager(8888, "pipe1");
            VCpuManager vcpu2 = new VCpuManager(8889, "pipe2");
            VCpuManager vcpu3 = new VCpuManager(8890, "pipe3 ");

            vcpu1.StartTcpNodeService();
            vcpu2.StartTcpNodeService();
            vcpu3.StartTcpNodeService();

            vcpu1.NetworkNodeManager.Register(vcpu2.GetMyNetworkNodeInfo());
            vcpu3.NetworkNodeManager.Register(vcpu1.GetMyNetworkNodeInfo());


            Assert.AreEqual(3, vcpu1.NetworkNodeManager.NodeService.KnownNodes.Count);
            Assert.AreEqual(3, vcpu2.NetworkNodeManager.NodeService.KnownNodes.Count);
            Assert.AreEqual(3, vcpu3.NetworkNodeManager.NodeService.KnownNodes.Count);

            Assert.AreEqual(2, vcpu1.NetworkNodeManager.NodeServiceClients.Count);
            Assert.AreEqual(2, vcpu2.NetworkNodeManager.NodeServiceClients.Count);
            Assert.AreEqual(2, vcpu3.NetworkNodeManager.NodeServiceClients.Count);

        }

        [TestMethod]
        public void ClusterCalculations()
        {
            VCpuManager vcpu1 = new VCpuManager(8888, "pipe1");
            VCpuManager vcpu2 = new VCpuManager(8889, "pipe2");
            VCpuManager vcpu3 = new VCpuManager(8890, "pipe3 ");

            vcpu1.StartTcpNodeService();
            vcpu2.StartTcpNodeService();
            vcpu3.StartTcpNodeService();

            vcpu1.NetworkNodeManager.Register(vcpu2.GetMyNetworkNodeInfo());
            vcpu3.NetworkNodeManager.Register(vcpu1.GetMyNetworkNodeInfo());
            var settings = PsoSettingsFactory.QuadraticFunction1DFrom3To5();
            settings.Iterations = 1000;
            settings.IterationsLimitCondition = true;
            vcpu1.StartCalculations(settings);
            PsoController contrl = new PsoController();
            contrl.Run(settings);

            var result = vcpu1.PsoController.RunningAlgorithm.Result;

            Assert.AreEqual(-9.0,result.FitnessValue,0.1);
        }

        [TestMethod]
        public void IdTest()
        {
            string ipString = "net.tcp://" + IPAddress.Loopback + ":8012/NodeService";   //"net.pipe://localhost/NodeService/" + pipeName)"

            string[] parts = ipString.Split('/');
            string[] iparts = parts[2].Split(':');

            Byte[] bytes = (IPAddress.Parse(iparts[0])).GetAddressBytes();

            ulong ip = (ulong)(BitConverter.ToInt32(bytes, 0));
            ulong port = (ulong)(Int32.Parse(iparts[1]));

            ulong id = (ip << 32) + port;

            var ipb = BitConverter.GetBytes(ip);
            var portb = BitConverter.GetBytes(port);
            var idb = BitConverter.GetBytes(id);

            return;
        }
    }
}
