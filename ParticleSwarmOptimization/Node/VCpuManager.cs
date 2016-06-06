using System;
using System.Collections.Generic;
using Common;
using Controller;
using NetworkManager;
using PsoService;

namespace Node
{
    public class VCpuManager
    {

        private NodeParameters _nodeParams;

        // GENERAL PART

        public VCpuManager(string tcpAddress,int tcpPort, string pipeName, IPsoController psoController  = null, IPsoManager psoRingManager = null)
        {
            NetworkNodeManager = new NetworkNodeManager(tcpAddress,tcpPort, pipeName);
            PsoController = psoController ?? new PsoController(NetworkNodeManager.NodeService.Info.Id);
            PsoRingManager = psoRingManager ?? new PsoRingManager(NetworkNodeManager.NodeService.Info.Id);
            NetworkNodeManager.NodeService.Info.ProxyParticlesAddresses = PsoRingManager.GetProxyParticlesAddresses();

            var uris = new List<Uri>();
            foreach (var address in NetworkNodeManager.NodeService.Info.ProxyParticlesAddresses)
            {
                var str = address.AbsoluteUri.Replace("0.0.0.0", tcpAddress);
                uris.Add(new Uri(str));
            }
            NetworkNodeManager.NodeService.Info.ProxyParticlesAddresses = uris.ToArray();

            NetworkNodeManager.NodeService.NeighborhoodChanged += PsoRingManager.UpdatePsoNeighborhood;
            NetworkNodeManager.NodeService.RegisterNode += RunOnNode;
            NetworkNodeManager.NodeService.StartCalculations += Run;
            PsoRingManager.CommunicationLost += NetworkNodeManager.NodeService.Deregister;
            PsoController.CalculationsCompleted += NetworkNodeManager.FinishCalculations;
            NetworkNodeManager.NodeService.RemoteCalculationsFinished += PsoController.RemoteControllerFinished;

        }

        // NETWORK PART

        public NetworkNodeManager NetworkNodeManager { get; set; }  // mo�na zrobi� z tego interfejs jak IPsoManager

        public void StartTcpNodeService()
        {
            NetworkNodeManager.StartTcpNodeService();
        }

        public void CloseTcpNodeService()
        {
            NetworkNodeManager.CloseTcpNodeService();
        }

        public void StartPipeNodeService()
        {
            NetworkNodeManager.StartPipeNodeService();
        }

        public void ClosePipeNodeService()
        {
            NetworkNodeManager.ClosePipeNodeService();
        }

        
        public NetworkNodeInfo GetMyNetworkNodeInfo()
        {
            return NetworkNodeManager.NodeService.Info;
        }
        public IPsoManager PsoRingManager { get; set; }
        public IPsoController PsoController { get; set; }

        public void Run( PsoParameters psoParameters)
        {
            if (!PsoController.CalculationsRunning)
            {
                PsoController.Run(psoParameters, PsoRingManager.GetProxyParticles());
            }
        }

        public void StartCalculations(PsoParameters parameters)
        {
            if (PsoController.CalculationsRunning) return;
            NetworkNodeManager.StartCalculations(parameters);
            Run(parameters);
        }

        private void RunOnNode(NetworkNodeInfo newNode)
        {
            if (PsoController.CalculationsRunning)
            {
                NetworkNodeManager.StartCalculations(PsoController.RunningParameters,newNode);
            }
        }
    }
}