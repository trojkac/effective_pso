using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.ServiceModel;
using Common;
using PsoService;

namespace NetworkManager
{
    [ServiceBehavior(InstanceContextMode = InstanceContextMode.Single)]
    public class NodeService : INodeService, IReconnaissance
    {
        public event UpdateNeighborhoodHandler NeighborhoodChangedEvent;
        public event StartCalculationsHandler StartCalculations;
        public NetworkNodeInfo Info { get; set; }
        public IPsoManager PsoManager { get; private set; }
        /// <summary>
        /// List of all nodes in cluster (including the current one).
        /// </summary>
        public List<NetworkNodeInfo> KnownNodes { get; private set; }


        public NodeService(string tcpAddress, int tcpPort, string pipeName)
            : this()
        {
            Info = new NetworkNodeInfo("net.tcp://" + tcpAddress + ":" + tcpPort + "/NodeService", "net.pipe://" + "127.0.0.1" + "/NodeService/" + pipeName);
            KnownNodes.Add(Info);
            PsoManager = new PsoRingManager(Info.Id);
        }
        private NodeService()
        {
            KnownNodes = new List<NetworkNodeInfo>();
        }


        public void UpdateNodes(NetworkNodeInfo[] nodes)
        {
            foreach (var networkNodeInfo in nodes)
            {
                if (KnownNodes.Contains(networkNodeInfo)) continue;
                Debug.WriteLine("{0}: updating nodes", Info.Id);
                KnownNodes = new List<NetworkNodeInfo>(nodes);
                if (NeighborhoodChangedEvent != null) NeighborhoodChangedEvent(KnownNodes.ToArray(), Info);
            }
        }

        public void Register(NetworkNodeInfo source)
        {
            Debug.WriteLine("{0}: registering new node", Info.Id);
            KnownNodes.Add(source);
            if (NeighborhoodChangedEvent != null) NeighborhoodChangedEvent(KnownNodes.ToArray(), Info);
            BroadcastNeighborhoodList();
        }

        public void StartCalculation(PsoSettings settings)
        {
            Debug.WriteLine("{0}: starting calculations.", Info.Id);
            if (StartCalculations != null) StartCalculations(settings);
        }


        private void BroadcastNeighborhoodList()
        {
            Debug.WriteLine("{0}: broadcasting neighbors list", Info.Id);
            foreach (var networkNodeInfo in KnownNodes)
            {
                if (networkNodeInfo.Id == Info.Id) continue;
                NodeServiceClient nodeServiceClient = new TcpNodeServiceClient(networkNodeInfo.TcpAddress);
                nodeServiceClient.UpdateNodes(KnownNodes.ToArray());
            }
        }

    }
}
