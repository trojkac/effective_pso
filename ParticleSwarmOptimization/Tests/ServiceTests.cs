using System;
using System.Collections.Generic;
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
            Uri uri1 = new Uri("net.tcp://localhost:1234/");
            Uri uri2 = new Uri("net.tcp//localhost:1235/");

            EndpointAddress endpointAddress1 = new EndpointAddress(uri1);
            EndpointAddress endpointAddress2 = new EndpointAddress(uri2);

            //NetworkNodeInfo nodeInfo1 = new NetworkNodeInfo(endpointAddress1);
            //NetworkNodeInfo nodeInfo2 = new NetworkNodeInfo(endpointAddress2);

            HashSet<NetworkNodeInfo> bootstrap = new HashSet<NetworkNodeInfo>();

            global::Node.Node node1 = new global::Node.Node(endpointAddress1);
            bootstrap.Add(node1.MyInfo);
            global::Node.Node node2 = new global::Node.Node(bootstrap, endpointAddress2);

            node1.StartP2PAlgorithm();
            node2.StartP2PAlgorithm();
        }
    }
}
