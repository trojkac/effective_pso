using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.ServiceModel;
using Common;
using System.Threading.Tasks;
using Common.Parameters;
using PsoService;

namespace NetworkManager
{
    [ServiceBehavior(InstanceContextMode = InstanceContextMode.Single)]
    public class NodeService : INodeService, IReconnaissance
    {
        
        public event UpdateNeighborhoodHandler NeighborhoodChanged;
        public event StartCalculationsHandler StartCalculations;
        public event RegisterNodeHandler RegisterNode;
        public event RemoteCalculationsFinishedHandler RemoteCalculationsFinished;
        public event StopCalculationsHandler StopCalculations;
        public NetworkNodeInfo Info { get; set; }
        public IPsoManager PsoManager { get; private set; }
        /// <summary>
        /// List of all nodes in cluster (including the current one).
        /// </summary>
        public List<NetworkNodeInfo> KnownNodes { get; set; }


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
                if (NeighborhoodChanged != null) NeighborhoodChanged(KnownNodes.ToArray(), Info);
            }
        }

        public NetworkNodeInfo[] Register(NetworkNodeInfo source)
        {
            Debug.WriteLine("{0}: registering new node", Info.Id);
            KnownNodes.Add(source);
            BroadcastNeighborhoodList(source);
            Task.Factory.StartNew(() =>
            {
                if (RegisterNode != null) RegisterNode(source);
                if (NeighborhoodChanged != null) NeighborhoodChanged(KnownNodes.ToArray(), Info);
            });
            return KnownNodes.ToArray();
            
        }

        public void Deregister(NetworkNodeInfo brokenNodeInfo)
        {
            var nodes = KnownNodes.Where(n => n.Id == brokenNodeInfo.Id);
            if (!nodes.Any())
            {
                Debug.WriteLine("{0}:  Trying to deregister node {1}, already removed", Info.Id, brokenNodeInfo.Id);
                return;
            }
            Debug.WriteLine("{0}: deregistering node {1}", Info.Id, brokenNodeInfo.Id);
            var node = KnownNodes.First(n => n.Id == brokenNodeInfo.Id);
            KnownNodes.Remove(node);
            BroadcastNeighborhoodList();
            Task.Factory.StartNew(() =>
            {
                if (NeighborhoodChanged != null) NeighborhoodChanged(KnownNodes.ToArray(), Info);
            });
        }

        public void StartCalculation(PsoParameters parameters, NetworkNodeInfo mainNodeInfo)
        {
            Debug.WriteLine("{0}: starting calculations.", Info.Id);
            if (StartCalculations != null) StartCalculations(parameters, mainNodeInfo);
        }

        public ParticleState StopCalculation()
        {
            var pState = new ParticleState();
            Debug.WriteLine("{0}: received stop calculations", Info.Id);
          try
          {
            if (StopCalculations != null) pState = StopCalculations();

          }
          catch (InvalidProgramException e)
          {
            Debug.WriteLine("{0}: received stop calculations but {1}", e.Message);
            return pState;
          }
            return pState;
        }

        public void CalculationsFinished(NetworkNodeInfo source, ParticleState result)
        {
            Debug.WriteLine("{0}: received calculations finish from {1}",Info.Id, source.Id);
            if (RemoteCalculationsFinished != null)
            {
                RemoteCalculationsFinished(new RemoteCalculationsFinishedHandlerArgs(source, result));
            }
        }

        public void CheckStatus()
        {
            
        }


        private void BroadcastNeighborhoodList(NetworkNodeInfo skipNode = null)
        {
            Debug.WriteLine("{0}: broadcasting neighbors list", Info.Id);
            foreach (var networkNodeInfo in KnownNodes)
            {
                if (networkNodeInfo.Id == Info.Id ||(skipNode != null && networkNodeInfo.Id == skipNode.Id)) continue;
                NodeServiceClient nodeServiceClient = new TcpNodeServiceClient(networkNodeInfo.TcpAddress);
                nodeServiceClient.UpdateNodes(KnownNodes.ToArray());
            }
        }
    }
}
