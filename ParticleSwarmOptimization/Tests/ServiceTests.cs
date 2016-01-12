using System;
using System.Net;
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


            Assert.AreEqual(2,vcpu1.NetworkNodeManager.NodeService.KnownNodes.Count);
            Assert.AreEqual(2, vcpu2.NetworkNodeManager.NodeService.KnownNodes.Count);

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
