﻿using System;
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
            VCpuManager vcpu1 = new VCpuManager("192.168.142.32",8888, "pipe1");
            VCpuManager vcpu2 = new VCpuManager("192.168.142.32", 8889, "pipe2");
            VCpuManager vcpu3 = new VCpuManager("192.168.142.32", 8890, "pipe3 ");

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
            VCpuManager vcpu1 = new VCpuManager("192.168.111.13", 8888, "pipe1");
            VCpuManager vcpu2 = new VCpuManager("192.168.111.13", 8889, "pipe2");
            VCpuManager vcpu3 = new VCpuManager("192.168.111.13", 8890, "pipe3 ");

            vcpu1.StartTcpNodeService();
            vcpu2.StartTcpNodeService();
            vcpu3.StartTcpNodeService();

            vcpu3.NetworkNodeManager.Register(new NetworkNodeInfo("net.tcp://192.168.111.13:8889/NodeService", ""));

            vcpu1.NetworkNodeManager.Register(new NetworkNodeInfo("net.tcp://192.168.111.13:8889/NodeService",""));
            var settings = PsoSettingsFactory.QuadraticFunction1DFrom3To5();
            settings.Dimensions = 20;
            settings.FunctionParameters.Dimension = settings.Dimensions;
            settings.Iterations = 1000;
            settings.IterationsLimitCondition = true;
            settings.FunctionParameters.SearchSpace = new Tuple<double, double>[settings.Dimensions];
            settings.FunctionParameters.Coefficients = new double[settings.Dimensions];
            for (int i = 0; i < settings.Dimensions; i++)
            {
                settings.FunctionParameters.SearchSpace[i] = new Tuple<double, double>(-4.0,4.0);
                settings.FunctionParameters.Coefficients[i] = 1;
            }
            vcpu1.StartCalculations(settings);
           
            var result = vcpu1.PsoController.RunningAlgorithm.Result;

            Assert.AreEqual(0.0,result.FitnessValue,0.1);
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
