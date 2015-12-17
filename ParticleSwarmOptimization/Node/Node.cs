using System.Collections.Generic;
using System.ServiceModel;
using Common;
using PsoService;

namespace Node
{
    public class Node
    {
        public NodeManager NodeManager { get; set; }
        public IPsoManager PsoRingManager { get; set; }

        public Node(NodeManager nodeManager, IPsoManager psoRingManager)
        {
            NodeManager = nodeManager;
            PsoRingManager = psoRingManager;
        }

        public Node(EndpointAddress endpointAddress)
        {
            NodeManager = new NodeManager(new NodeService(endpointAddress));
        }

        public Node(HashSet<NetworkNodeInfo> bootstrap, EndpointAddress endpointAddress)
        {
            NodeManager = new NodeManager(new NodeService(bootstrap, endpointAddress));
        }

        public Node(HashSet<NetworkNodeInfo> bootstrap, NetworkNodeInfo myInfo)
        {
            NodeManager = new NodeManager(new NodeService(bootstrap, myInfo));
        }

        public void StartNodeService()
        {
            NodeManager.StartNodeService();
        }

        public void CloseNodeService()
        {
            NodeManager.CloseNodeService();
        }

        public NetworkNodeInfo GetMyNetworkNodeInfo()
        {
            return NodeManager.NodeService.MyInfo;
        }
    }
}