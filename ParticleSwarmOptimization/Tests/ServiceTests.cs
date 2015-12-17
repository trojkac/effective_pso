using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.ServiceModel;
using Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;


namespace Tests
{
    [TestClass]
    public class ServiceTests
    {
        [TestMethod]
        public void TestMethod1()
        {
            Uri uri1 = new Uri("net.tcp://localhost:1234");
            Uri uri2 = new Uri("net.tcp://localhost:1235");

            EndpointAddress endpointAddress1 = new EndpointAddress(uri1);
            EndpointAddress endpointAddress2 = new EndpointAddress(uri2);

            HashSet<NetworkNodeInfo> bootstrap = new HashSet<NetworkNodeInfo>();

            var node1 = new Node.Node(endpointAddress1);
            bootstrap.Add(node1.GetMyNetworkNodeInfo());
            var node2 = new Node.Node(bootstrap, endpointAddress2);

            node1.StartNodeService();
            node2.StartNodeService();

            Console.ReadKey();
        }
    }
}
