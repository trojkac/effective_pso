using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Node;


namespace Tests
{
    [TestClass]
    public class ServiceTests
    {
        [TestMethod]
        public void TestMethod1()
        {
            VCpuManager vcpu1 = new VCpuManager(8888, "pipe1");
            VCpuManager vcpu2 = new VCpuManager(8889, "pipe2");

            vcpu1.StartTcpNodeService();
            vcpu2.StartTcpNodeService();

            vcpu1.NetworkNodeManager.AddBootstrappingPeer(vcpu2.GetMyNetworkNodeInfo());
            vcpu2.NetworkNodeManager.AddBootstrappingPeer(vcpu1.GetMyNetworkNodeInfo());

            Console.ReadKey();
        }
    }
}
