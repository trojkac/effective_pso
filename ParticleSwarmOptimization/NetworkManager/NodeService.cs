using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.ServiceModel;

namespace NetworkManager
{
    [ServiceBehavior(InstanceContextMode = InstanceContextMode.Single)]
    public class NodeService : INodeService
    {
        private System.Object _lockObject = new System.Object();

        public NetworkNodeInfo Info { get; set; }

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

        private readonly List<NetworkNodeInfo> _neighbors; //Gamma   //czy nie trzeba uważać na przypadek, gdy to jest puste?

        public NodeService()
        {
            _peers = new HashSet<NetworkNodeInfo>();
            _searchMonitorNodes = new HashSet<NetworkNodeInfo>();
            _closerPeerSearchNodes = new HashSet<NetworkNodeInfo>();
            _neighbors = new List<NetworkNodeInfo>();
            BootstrappingPeers = new HashSet<NetworkNodeInfo>();
        }

        public NodeService(int tcpPort, string pipeName)
            : this()
        {
            Info = new NetworkNodeInfo("net.tcp://localhost:" + tcpPort + "/NodeService", "net.pipe://localhost/NodeService/" + pipeName);
        }

        public NodeService(string tcpAddress, string pipeAddress)
            : this()
        {
            Info = new NetworkNodeInfo(tcpAddress, pipeAddress);
        }

        public List<NetworkNodeInfo> GetNeighbors()
        {
            return _neighbors;
        }

        public void AddBootstrappingPeer(NetworkNodeInfo peerInfo)
        {
            BootstrappingPeers.Add(peerInfo);
        }

        public void SetBootstrappingPeers(HashSet<NetworkNodeInfo> bootstrappingPeers)
        {
            BootstrappingPeers = bootstrappingPeers;
        }

        public NetworkNodeInfo GetClosestNeighbor(NetworkNodeInfo x)  //what if it returns null?
        {
            int minDistance = Int32.MaxValue;
            NetworkNodeInfo closestNeighbor = null;

            foreach (NetworkNodeInfo nodeInfo in _neighbors)
            {
                if (NetworkNodeInfo.Distance(nodeInfo, x) < minDistance)
                {
                    minDistance = NetworkNodeInfo.Distance(nodeInfo, x);
                    closestNeighbor = nodeInfo;
                }
            }

            return closestNeighbor;
        }

        public void A1()
        {
            Debug.WriteLine("NodeService o adresie: " + Info.TcpAddress + " wykonuje A1()");

            lock (_lockObject)
            {
                if (_s != null) _peers.Add(_s);
                _peers.UnionWith(_closerPeerSearchNodes);
                _peers.UnionWith(_searchMonitorNodes);
                _peers.UnionWith(_neighbors);


                if (GetClosestNeighbor(Info) != null)
                {
                    _neighbors.Insert(0, GetClosestNeighbor(Info));
                }
                _searchMonitorNodes.Clear();
                _closerPeerSearchNodes.Clear();
            }
        }

        public void A2()
        {
            Debug.WriteLine("NodeService o adresie: " + Info.TcpAddress + " wykonuje A2()");

            lock (_lockObject)
            {
                Random random = new Random(); //do klasy?
                int r = random.Next(0, _neighbors.Count > 0 ? BootstrappingPeers.Count + 1 : BootstrappingPeers.Count);
                if (_neighbors.Count == 0 && BootstrappingPeers.Count == 0)
                {
                    return;
                }

                _s = r < BootstrappingPeers.Count ? BootstrappingPeers.ElementAt(r) : _neighbors[0];

                NodeServiceClient nodeServiceClient = new TcpNodeServiceClient(_s.TcpAddress);
                nodeServiceClient.CloserPeerSearch(Info);
            }
        }

        public void A5()
        {
            Debug.WriteLine("NodeService o adresie: " + Info.TcpAddress + " wykonuje A5()");

            lock (_lockObject)
            {
                for (int i = 0; i < _neighbors.Count; ++i)
                {
                    NodeServiceClient nodeServiceClient = new TcpNodeServiceClient(_neighbors[i].TcpAddress);
                    nodeServiceClient.GetNeighbor(Info, i);
                }
            }
        }

        public void CloserPeerSearch(NetworkNodeInfo source) //A3
        {
            Debug.WriteLine("NodeService o adresie: " + source.TcpAddress + " wywołuje A3() na serwisie o adresie: " + Info.TcpAddress);

            lock (_lockObject)
            {
                if (_neighbors.Count == 0)
                {
                    _searchMonitorNodes.Add(source);
                }
                else
                {
                    if (NetworkNodeInfo.Distance(Info, source) <
                        _neighbors.Min(n => NetworkNodeInfo.Distance(n, source)))
                    {
                        _searchMonitorNodes.Add(source);
                        NodeServiceClient nodeServiceClient = new TcpNodeServiceClient(source.TcpAddress);
                        nodeServiceClient.SuccessorCandidate(_neighbors[0]);
                    }
                    else
                    {
                        NodeServiceClient nodeServiceClient =
                            new TcpNodeServiceClient(GetClosestNeighbor(source).TcpAddress);
                        nodeServiceClient.CloserPeerSearch(source);
                    }
                }
            }
        }

        public void SuccessorCandidate(NetworkNodeInfo candidate) //A4
        {
            Debug.WriteLine("NodeService o adresie: " + candidate.TcpAddress + " wywołuje A4() na serwisie o adresie: " + Info.TcpAddress);

            lock (_lockObject)
            {
                _closerPeerSearchNodes.Add(candidate);
            }
        }

        public void GetNeighbor(NetworkNodeInfo from, int j) //A6
        {
            Debug.WriteLine("NodeService o adresie: " + from.TcpAddress + " wywołuje A3() na serwisie o adresie: " + Info.TcpAddress);

            lock (_lockObject)
            {
                if (_neighbors.Count > j)
                {
                    NodeServiceClient nodeServiceClient = new TcpNodeServiceClient(from.TcpAddress);
                    nodeServiceClient.UpdateNeighbor(_neighbors[j], j);
                }
            }
        }

        public void UpdateNeighbor(NetworkNodeInfo newNeighbor, int c) //A7
        {
            Debug.WriteLine("NodeService o adresie: " + newNeighbor.TcpAddress + " wywołuje A7() na serwisie o adresie: " + Info.TcpAddress);

            lock (_lockObject)
            {
                if (NetworkNodeInfo.Distance(_neighbors[c], newNeighbor) <
                    NetworkNodeInfo.Distance(_neighbors[c], Info)) //is it ok?
                {
                    _neighbors[c + 1] = newNeighbor;
                }
                else
                {
                    _neighbors.RemoveAt(c + 1);
                }
            }
        }
    }
}
