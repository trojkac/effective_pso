using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
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

        public VCpuManager(string tcpAddress,int tcpPort, string pipeName, IPsoController psoController  = null, IPsoManager psoRingManager = null)
        {
            NetworkNodeManager = new NetworkNodeManager(tcpAddress,tcpPort, pipeName);
            PsoController = psoController ?? new PsoController();
            PsoRingManager = psoRingManager ?? new PsoRingManager(NetworkNodeManager.NodeService.Info.Id);
            NetworkNodeManager.NodeService.Info.ProxyParticlesAddresses = PsoRingManager.GetProxyParticlesAddresses();

            var uris = new List<Uri>();
            foreach (var address in NetworkNodeManager.NodeService.Info.ProxyParticlesAddresses)
            {
                var str = address.AbsoluteUri.Replace("0.0.0.0", tcpAddress);
                uris.Add(new Uri(str));
            }
            NetworkNodeManager.NodeService.Info.ProxyParticlesAddresses = uris.ToArray();

            NetworkNodeManager.NodeService.NeighborhoodChangedEvent += PsoRingManager.UpdatePsoNeighborhood;
            NetworkNodeManager.NodeService.NeighborhoodChangedEvent += RunOthers;
            NetworkNodeManager.NodeService.StartCalculations += Run;
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

        public void Run( PsoSettings psoSettings)
        {
            if (!PsoController.CalculationsRunning)
            {
                PsoController.Run(psoSettings, PsoRingManager.GetProxyParticles());
            }
        }

        public void StartCalculations(PsoSettings settings)
        {
            if (PsoController.CalculationsRunning) throw new NotSupportedException("Calculations are already running.");
            NetworkNodeManager.StartCalculations(settings);
            Run(settings);
        }

        private void RunOthers(NetworkNodeInfo[] allNetworkNodes, NetworkNodeInfo currentNetworkNode)
        {
            if (PsoController.CalculationsRunning)
            {
                NetworkNodeManager.StartCalculations(PsoController.RunningSettings);

            }
        }
    }
}