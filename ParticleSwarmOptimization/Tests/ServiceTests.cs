using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.ServiceModel;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Node;
using Node = Node.Node;

namespace Tests
{
    [TestClass]
    public class ServiceTests
    {
        [TestMethod]
        public void TestMethod1()
        {
            Uri uri1 = new Uri("net.tcp://localhost:1234//NodeService");
            Uri uri2 = new Uri("net.tcp://localhost:1235//NodeService");

            EndpointAddress endpointAddress1 = new EndpointAddress(uri1);
            EndpointAddress endpointAddress2 = new EndpointAddress(uri2);

            HashSet<NetworkNodeInfo> bootstrap = new HashSet<NetworkNodeInfo>();

            global::Node.Node node1 = new global::Node.Node(endpointAddress1);
            bootstrap.Add(node1.GetMyNetworkNodeInfo());
            global::Node.Node node2 = new global::Node.Node(bootstrap, endpointAddress2);

            node1.StartNodeService();
            node2.StartNodeService();

            Console.ReadKey();
        }
    }
}
