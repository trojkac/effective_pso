using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.ServiceModel;
using Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NetworkManager;
using Node;


namespace Tests
{
    [TestClass]
    public class ServiceTests
    {
        [TestMethod]
        public void TestMethod1()
        {
            //Uri uri1 = new Uri("net.tcp://localhost:7777");
            //Uri uri2 = new Uri("net.tcp://localhost:7778");

            //EndpointAddress endpointAddress1 = new EndpointAddress(uri1);
            //EndpointAddress endpointAddress2 = new EndpointAddress(uri2);

            //HashSet<NetworkNodeInfo> bootstrap = new HashSet<NetworkNodeInfo>();

            //var node1 = new VCpuManager.VCpuManager(endpointAddress1);
            //bootstrap.Add(node1.GetMyNetworkNodeInfo());
            //var node2 = new VCpuManager.VCpuManager(bootstrap, endpointAddress2);

            //node1.StartNodeService();
            //Debug.WriteLine("node1.StartNodeService()");
            //node2.StartNodeService();

            //Console.ReadKey();

            VCpuManager vcpu1 = new VCpuManager(8888, "pipe1");
            VCpuManager vcpu2 = new VCpuManager(8889, "pipe2");

            vcpu1.StartTcpNodeService();
            vcpu2.StartTcpNodeService();

            vcpu1.NetworkNodeManager.AddBootstrappingPeer(vcpu2.GetMyNetworkNodeInfo());
            vcpu2.NetworkNodeManager.AddBootstrappingPeer(vcpu2.GetMyNetworkNodeInfo());
        }
    }
}
