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
        private Random random = new Random();

        public NetworkNodeInfo Info { get; set; }
        public NetworkNodeInfo Successor { get { return _neighbors[0]; } }

        //CONST
        public HashSet<NetworkNodeInfo> BootstrappingPeers { get; set; }

        //INPUT
        private NetworkNodeInfo _s;

        //VAR
        private readonly HashSet<NetworkNodeInfo> _peers; //S
        private readonly HashSet<NetworkNodeInfo> _successorCandidatesFromSearchMonitor; //B
        private readonly HashSet<NetworkNodeInfo> _successorCandidatesFormCloserPeerSearch; //W

        private readonly List<NetworkNodeInfo> _neighbors; //Gamma   //czy nie trzeba uważać na przypadek, gdy to jest puste?  //zakładam, że węzeł nie może mieć wśród swoich sąsiadów siebie

        public NodeService()
        {
            _peers = new HashSet<NetworkNodeInfo>();
            _successorCandidatesFromSearchMonitor = new HashSet<NetworkNodeInfo>();
            _successorCandidatesFormCloserPeerSearch = new HashSet<NetworkNodeInfo>();
            _neighbors = new List<NetworkNodeInfo>();
            BootstrappingPeers = new HashSet<NetworkNodeInfo>();
        }

        public NodeService(int tcpPort, string pipeName)
            : this()
        {
            Info = new NetworkNodeInfo("net.tcp://" + "127.0.0.1" + ":" + tcpPort + "/NodeService", "net.pipe://" + "127.0.0.1" + "/NodeService/" + pipeName);
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

        //Neighbor Update part I - aktualizacja następnika
        public void A1()
        {
            Debug.WriteLine("NodeService o adresie TCP: " + Info.TcpAddress + " wykonuje Neighbor Update part I");

            lock (_lockObject)
            {
                _peers.Clear();
                if (_s != null) { _peers.Add(_s); }
                _peers.UnionWith(_successorCandidatesFormCloserPeerSearch);
                _peers.UnionWith(_successorCandidatesFromSearchMonitor);
                _peers.UnionWith(_neighbors);

                _peers.Remove(Info);

                NetworkNodeInfo closestPeer = NetworkNodeInfo.GetBestSuccessorCandidate(Info, _peers);
                if (closestPeer != null)
                {
                    _neighbors.Remove(closestPeer);
                    _neighbors.Insert(0, closestPeer);
                }
                _successorCandidatesFromSearchMonitor.Clear();
                _successorCandidatesFormCloserPeerSearch.Clear();
            }
        }

        //sending Closer-Peer Search from Info
        public void A2()
        {
            Debug.WriteLine("NodeService o adresie TCP: " + Info.TcpAddress + " inicjuje Closer-Peer Search");

            lock (_lockObject)
            {
                int r = random.Next(0, _neighbors.Count > 0 ? BootstrappingPeers.Count + 1 : BootstrappingPeers.Count);
                if (_neighbors.Count == 0 && BootstrappingPeers.Count == 0)
                {
                    return;
                }

                _s = r < BootstrappingPeers.Count ? BootstrappingPeers.ElementAt(r) : _neighbors[0];

                if (!_s.Equals(Info))
                {
                    try
                    {
                        NodeServiceClient nodeServiceClient = new TcpNodeServiceClient(_s.TcpAddress);
                        nodeServiceClient.CloserPeerSearch(Info);
                    }
                    catch (Exception e) { }
                }
            }
        }

        //receiving Closer-Peer Search from 'source'
        public void CloserPeerSearch(NetworkNodeInfo source) //A3
        {
            Debug.WriteLine("NodeService o adresie TCP: " + Info.TcpAddress + " otrzymał zapytanie Closer-Peer Search od serwisu o adresie TCP: " + source.TcpAddress);

            lock (_lockObject)
            {
                if (source.Equals(Info))
                {
                    return;
                }

                if (_neighbors.Count == 0)
                {
                    _successorCandidatesFromSearchMonitor.Add(source);
                }
                else
                {
                    if (NetworkNodeInfo.Distance(Info, source) <=
                        _neighbors.Min(n => NetworkNodeInfo.Distance(n, source)))
                    {
                        _successorCandidatesFromSearchMonitor.Add(source);

                        try
                        {
                            NodeServiceClient nodeServiceClient = new TcpNodeServiceClient(source.TcpAddress);
                            nodeServiceClient.SuccessorCandidate(_neighbors[0]);
                        }
                        catch (Exception e) { }
                    }
                    else
                    {
                        try
                        {
                            NodeServiceClient nodeServiceClient =
                                new TcpNodeServiceClient(NetworkNodeInfo.GetClosestPeer(source, _neighbors).TcpAddress);
                            nodeServiceClient.CloserPeerSearch(source);
                        }
                        catch (Exception e) { }
                    }
                }
            }
        }

        //receiving SuccessorCandidate - 'candidate'
        public void SuccessorCandidate(NetworkNodeInfo candidate) //A4
        {
            Debug.WriteLine("NodeService o adresie TCP: " + Info.TcpAddress + " otrzymał successorCandidate o adresie TCP: " + candidate.TcpAddress);

            lock (_lockObject)
            {
                _successorCandidatesFormCloserPeerSearch.Add(candidate);  //nie trzeba tu sprawdzać, czy nie dodajemy Info, bo to jest robione w A1
            }
        }

        //initiating Neighbor Update part II - próba poprawienia sąsiadów o większych numerach
        public void A5()
        {
            Debug.WriteLine("NodeService o adresie TCP: " + Info.TcpAddress + " wykonuje Neighbor Update part II");

            lock (_lockObject)
            {
                for (int i = 1; i < _neighbors.Count; ++i)
                {
                    try
                    {
                        NodeServiceClient nodeServiceClient = new TcpNodeServiceClient(_neighbors[i].TcpAddress);
                        nodeServiceClient.GetNeighbor(Info, i);
                    }
                    catch (Exception e) { }
                }
            }
        }

        //receiving Neighbor Update part II
        public void GetNeighbor(NetworkNodeInfo from, int j) //A6
        {
            Debug.WriteLine("NodeService o adresie TCP: " + Info.TcpAddress + " otrzymał zapytanie o sąsiada nr " + j + " od serwisu o adresie TCP: " + from.TcpAddress);

            lock (_lockObject)
            {
                if (_neighbors.Count > j)
                {
                    try
                    {
                        NodeServiceClient nodeServiceClient = new TcpNodeServiceClient(from.TcpAddress);
                        nodeServiceClient.UpdateNeighbor(_neighbors[j], j);
                    }
                    catch (Exception e) { }
                }
            }
        }

        //receiving result of Neighbor Update part II
        public void UpdateNeighbor(NetworkNodeInfo potentialNeighbor, int c) //A7
        {
            Debug.WriteLine("NodeService o adresie TCP: " + Info.TcpAddress + " otrzymał kandydata na sąsiada nr " + c + " o adresie TCP: " + potentialNeighbor.TcpAddress);

            lock (_lockObject)
            {
                if (NetworkNodeInfo.IsBetween(_neighbors[c], Info, potentialNeighbor) && !_neighbors.Contains(potentialNeighbor))
                {
                    _neighbors[c + 1] = potentialNeighbor;
                }
                //else
                //{
                //    if (_neighbors.Count > c + 1)
                //    {
                //        _neighbors.RemoveAt(c + 1); // co znaczy _neighbors[c+1] = NIL?
                //    }
                //}
            }
        }
    }
}
