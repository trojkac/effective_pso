using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Remoting.Messaging;
using System.ServiceModel;
using Common;

namespace Node
{
    [ServiceBehavior(InstanceContextMode = InstanceContextMode.Single)]
    public class NodeService : INodeService
    {
        public NetworkNodeInfo MyInfo { get; set; }

        //CONST
        public HashSet<NetworkNodeInfo> BootstrappingPeers { get; set; }

        //INPUT
        //private NetworkNodeInfo _w;
        //private NetworkNodeInfo _x;
        //private int _c;
        //private NetworkNodeInfo _z;
        private NetworkNodeInfo _s;

        //VAR
        private readonly HashSet<NetworkNodeInfo> _peers; //S
        private readonly HashSet<NetworkNodeInfo> _searchMonitorNodes; //B
        private readonly HashSet<NetworkNodeInfo> _closerPeerSearchNodes; //W

        private readonly List<NetworkNodeInfo> _neighbors;
        //Gamma   //czy nie trzeba uważać na przypadek, gdy to jest puste?

        public NodeService(EndpointAddress endpointAddress)
        {
            MyInfo = new NetworkNodeInfo(endpointAddress);
            BootstrappingPeers = new HashSet<NetworkNodeInfo>();
            _peers = new HashSet<NetworkNodeInfo>();
            _searchMonitorNodes = new HashSet<NetworkNodeInfo>();
            _closerPeerSearchNodes = new HashSet<NetworkNodeInfo>();
            _neighbors = new List<NetworkNodeInfo>();
        }

        public NodeService(HashSet<NetworkNodeInfo> bootstrap, EndpointAddress endpointAddress)
        {
            BootstrappingPeers = bootstrap;
            MyInfo = new NetworkNodeInfo(endpointAddress);
            _peers = new HashSet<NetworkNodeInfo>();
            _searchMonitorNodes = new HashSet<NetworkNodeInfo>();
            _closerPeerSearchNodes = new HashSet<NetworkNodeInfo>();
            _neighbors = new List<NetworkNodeInfo>();
        }

        public NodeService(HashSet<NetworkNodeInfo> bootstrap, NetworkNodeInfo myInfo)
        {
            BootstrappingPeers = bootstrap;
            MyInfo = myInfo;
            _peers = new HashSet<NetworkNodeInfo>();
            _searchMonitorNodes = new HashSet<NetworkNodeInfo>();
            _closerPeerSearchNodes = new HashSet<NetworkNodeInfo>();
            _neighbors = new List<NetworkNodeInfo>();
        }

        public NetworkNodeInfo GetClosestNeighbor()
        {
            int minDistance = Int32.MaxValue;
            NetworkNodeInfo closestNeighbor = null;
            foreach (NetworkNodeInfo nodeInfo in _peers)
            {
                if (NetworkNodeInfo.Distance(MyInfo, nodeInfo) < minDistance)
                {
                    minDistance = NetworkNodeInfo.Distance(MyInfo, nodeInfo);
                    closestNeighbor = nodeInfo;
                }
            }
            return closestNeighbor;
        }

        public void A1()
        {
            if (_s != null) _peers.Add(_s);
            _peers.UnionWith(_closerPeerSearchNodes);
            _peers.UnionWith(_searchMonitorNodes);
            _peers.UnionWith(_neighbors);

            _neighbors.Insert(0, GetClosestNeighbor());
            _searchMonitorNodes.Clear();
            _closerPeerSearchNodes.Clear();
        }

        public void A2()
        {
            Random random = new Random(); //do klasy?
            int r = random.Next(0, _neighbors.Count > 0 ? BootstrappingPeers.Count + 1 : BootstrappingPeers.Count);
            _s = r < BootstrappingPeers.Count ? BootstrappingPeers.ElementAt(r) : _neighbors[0];

            NodeServiceClient nodeServiceClient = new NodeServiceClient(_s);
            nodeServiceClient.CloserPeerSearch(MyInfo);
        }

        public void A5()
        {
            for (int i = 0; i < _neighbors.Count; ++i)
            {
                NodeServiceClient nodeServiceClient = new NodeServiceClient(_neighbors[i]);
                nodeServiceClient.GetNeighbor(MyInfo, i);
            }
        }

        public void CloserPeerSearch(NetworkNodeInfo source) //A3
        {
            if (NetworkNodeInfo.Distance(MyInfo, source) < _neighbors.Min(n => NetworkNodeInfo.Distance(MyInfo, n)))
            {
                _searchMonitorNodes.Add(source);
                NodeServiceClient nodeServiceClient = new NodeServiceClient(source);
                nodeServiceClient.SuccessorCandidate(_neighbors[0]);
            }
            else
            {
                NodeServiceClient nodeServiceClient = new NodeServiceClient(GetClosestNeighbor());
                nodeServiceClient.CloserPeerSearch(source);
            }
        }

        public void SuccessorCandidate(NetworkNodeInfo candidate) //A4
        {
            _closerPeerSearchNodes.Add(candidate);
        }

        public void GetNeighbor(NetworkNodeInfo from, int j) //A6
        {
            if (_neighbors.Count > j)
            {
                NodeServiceClient nodeServiceClient = new NodeServiceClient(from);
                nodeServiceClient.UpdateNeighbor(_neighbors[j], j);
            }
        }

        public void UpdateNeighbor(NetworkNodeInfo newNeighbor, int c) //A7
        {
            if (NetworkNodeInfo.Distance(_neighbors[c], newNeighbor) <
                NetworkNodeInfo.Distance(_neighbors[c], MyInfo)) //is it ok?
            {
                _neighbors[c + 1] = newNeighbor;
            } //what if not?
        }
    }
}
