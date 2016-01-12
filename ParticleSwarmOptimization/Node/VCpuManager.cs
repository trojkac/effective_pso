using System;
using System.Diagnostics;
using System.Threading;
using Common;
using NetworkManager;
using PsoService;
using Controller;

namespace Node
{
    public class VCpuManager
    {

        private UserNodeParameters _nodeParams;

        // GENERAL PART

        public VCpuManager(int tcpPort, string pipeName, IPsoController psoController  = null, IPsoManager psoRingManager = null)
        {
            NetworkNodeManager = new NetworkNodeManager(tcpPort, pipeName);
            PsoController = psoController ?? new PsoController();
            PsoRingManager = psoRingManager ?? new PsoRingManager(NetworkNodeManager.NodeService.Info.Id);
            NetworkNodeManager.NodeService.Info.ProxyParticlesAddresses = PsoRingManager.GetProxyParticlesAddresses();
            NetworkNodeManager.NodeService.NeighborhoodChangedEvent += PsoRingManager.UpdatePsoNeighborhood;
        }

        // NETWORK PART

        public NetworkNodeManager NetworkNodeManager { get; set; }  // mo¿na zrobiæ z tego interfejs jak IPsoManager

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

        public void Run(FitnessFunction fitnessFunction, PsoSettings psoSettings)
        {
            PsoController.Run(fitnessFunction, psoSettings, PsoRingManager.GetProxyParticles());
        }
    }
}