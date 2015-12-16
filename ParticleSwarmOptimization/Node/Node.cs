using System.Collections.Generic;
using System.ServiceModel;

namespace Node
{
    public class Node
    {
        public NodeManager NodeManager { get; set; }
        public PsoRingManager PsoRingManager { get; set; }

        public Node(NodeManager nodeManager, PsoRingManager psoRingManager)
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