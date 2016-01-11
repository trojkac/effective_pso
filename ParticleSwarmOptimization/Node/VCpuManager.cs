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
        private const int UpdateNeighborhoodMilis = 1024;
        private Timer _timer;

        private UserNodeParameters _nodeParams;

        //public VCpuManager(NetworkNodeManager networkNodeManager, IPsoManager psoRingManager)
        //{
        //    NetworkNodeManager = networkNodeManager;
        //    PsoRingManager = psoRingManager;
        //}

        //public VCpuManager(EndpointAddress endpointAddress)
        //{
        //    NetworkNodeManager = new NetworkNodeManager(new NodeService(endpointAddress));
        //}

        //public VCpuManager(HashSet<NetworkNodeInfo> bootstrap, EndpointAddress endpointAddress)
        //{
        //    NetworkNodeManager = new NetworkNodeManager(new NodeService(bootstrap, endpointAddress));
        //}

        //public VCpuManager(HashSet<NetworkNodeInfo> bootstrap, NetworkNodeInfo myInfo)
        //{
        //    NetworkNodeManager = new NetworkNodeManager(new NodeService(bootstrap, myInfo));
        //}

        // GENERAL PART

        public VCpuManager()
        {
        }

        public VCpuManager(int tcpPort, string pipeName)
        {
            NetworkNodeManager = new NetworkNodeManager(tcpPort, pipeName);
            PsoController = new PsoController();
            PsoRingManager = new PsoRingManager(NetworkNodeManager.NodeService.Info.Id);
            NetworkNodeManager.AddIPsoManager(PsoRingManager);
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

        public void StartPeriodicallyUpdatingNeighborhood()
        {
            TimerCallback timerCallback = UpdateNeighborhood;
            _timer = new Timer(timerCallback, null, UpdateNeighborhoodMilis, Timeout.Infinite);
        }

        public void UpdateNeighborhood(Object stateInfo)
        {
            Debug.WriteLine("Node " + NetworkNodeManager.NodeService.Info.Id + " rozpoczyna UpdateNeighborhood");
            PsoRingManager.UpdatePsoNeighborhood(NetworkNodeManager.GetAllProxyParticlesAddresses(), NetworkNodeManager.NodeService.Info);
            _timer.Change(UpdateNeighborhoodMilis, Timeout.Infinite);
        }

        public NetworkNodeInfo GetMyNetworkNodeInfo()
        {
            return NetworkNodeManager.NodeService.Info;
        }

        public void AddBootstrappingPeer(NetworkNodeInfo peer)
        {
            NetworkNodeManager.NodeService.AddBootstrappingPeer(peer);
        }

        // PSO PART
        public IPsoController PsoController { get; set; }

        public IPsoManager PsoRingManager { get; set; }
    }
}