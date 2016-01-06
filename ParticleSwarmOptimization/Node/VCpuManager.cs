using NetworkManager;
using PsoService;
using Controller;

namespace Node
{
    public class VCpuManager
    {
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
        }

        // NETWORK PART

        public NetworkNodeManager NetworkNodeManager { get; set; }  // mo¿e zrobiæ z tego interfejs jak IPsoManager?

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

        // PSO PART
        public IPsoController PsoController { get; set; }

        public IPsoManager PsoRingManager { get; set; }
    }
}