using System;
using System.Net;
using Common;
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
            VCpuManager vcpu1 = new VCpuManager("192.168.110.194", 8888, "pipe1");
            VCpuManager vcpu2 = new VCpuManager("192.168.110.194", 8889, "pipe2");
            VCpuManager vcpu3 = new VCpuManager("192.168.110.194", 8890, "pipe3 ");

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
            int cpuCores = 3;
            VCpuManager[] vcpus = new VCpuManager[cpuCores];
            for (int i = 0; i < cpuCores; i++)
            {
                vcpus[i] = new VCpuManager("192.168.110.194", 8881 + i, i.ToString());
                vcpus[i].StartTcpNodeService();
                if (i > 0)
                {
                    vcpus[i].NetworkNodeManager.Register(vcpus[i-1].GetMyNetworkNodeInfo());
                }
            }


            var t = System.Threading.Tasks.Task.Delay(1000);
            t.Wait();
            var settings = PsoSettingsFactory.QuadraticFunction20D();
            vcpus[0].StartCalculations(settings);


            var task = vcpus[0].PsoController.RunningAlgorithm;
            var result = task.Result;
            Assert.AreEqual(0.0, result.FitnessValue[0], 0.1);
        }

        [TestMethod]
        public void ClusterCalculations2()
        {
            int cpuCores = 8;
            VCpuManager[] vcpus = new VCpuManager[cpuCores];
            for (int i = 0; i < cpuCores; i++)
            {
                vcpus[i] = new VCpuManager("127.0.0.1", 8881 + i, i.ToString());
                vcpus[i].StartTcpNodeService();
                vcpus[i].NetworkNodeManager.Register(new NetworkNodeInfo("net.tcp://127.0.0.1:8881/NodeService", ""));
            }
            Console.ReadKey();
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
        }
    }
}
