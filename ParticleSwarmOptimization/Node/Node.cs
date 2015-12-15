using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Remoting.Messaging;
using System.ServiceModel;
using System.ServiceModel.Channels;
using System.Threading;

namespace Node
{
    [ServiceBehavior(InstanceContextMode = InstanceContextMode.Single)]
    public class Node : INodeService
    {
        private const int Miliseconds = 1000;
        private Timer _timer;

        public NetworkNodeInfo MyInfo { get; set; }
        private PsoRingManager _psoRingManager;

        //CONST
        public HashSet<NetworkNodeInfo> BootstrappingPeers { get; set; }

        //INPUT
        //private NetworkNodeInfo _w;
        //private NetworkNodeInfo _x;
        //private int _c;
        //private NetworkNodeInfo _z;
        private NetworkNodeInfo _s;

        //VAR
        private readonly HashSet<NetworkNodeInfo> _peers;  //S
        private readonly HashSet<NetworkNodeInfo> _searchMonitorNodes;  //B
        private readonly HashSet<NetworkNodeInfo> _closerPeerSearchNodes;  //W
        private readonly List<NetworkNodeInfo> _neighbors;  //Gamma   //czy nie trzeba uważać na przypadek, gdy to jest puste?

        public Node(EndpointAddress endpointAddress)
        {
            MyInfo = new NetworkNodeInfo(endpointAddress);
            BootstrappingPeers = new HashSet<NetworkNodeInfo>();
            _peers = new HashSet<NetworkNodeInfo>();
            _searchMonitorNodes = new HashSet<NetworkNodeInfo>();
            _closerPeerSearchNodes = new HashSet<NetworkNodeInfo>();
            _neighbors = new List<NetworkNodeInfo>();
            _psoRingManager = new PsoRingManager();
        }

        public Node(HashSet<NetworkNodeInfo> bootstrap, EndpointAddress endpointAddress)
        {
            BootstrappingPeers = bootstrap;
            MyInfo = new NetworkNodeInfo(endpointAddress);
            _peers = new HashSet<NetworkNodeInfo>();
            _searchMonitorNodes = new HashSet<NetworkNodeInfo>();
            _closerPeerSearchNodes = new HashSet<NetworkNodeInfo>();
            _neighbors = new List<NetworkNodeInfo>();
            _psoRingManager = new PsoRingManager();
        }

        public Node(HashSet<NetworkNodeInfo> bootstrap, NetworkNodeInfo myInfo)
        {
            BootstrappingPeers = bootstrap;
            MyInfo = myInfo;
            _peers = new HashSet<NetworkNodeInfo>();
            _searchMonitorNodes = new HashSet<NetworkNodeInfo>();
            _closerPeerSearchNodes = new HashSet<NetworkNodeInfo>();
            _neighbors = new List<NetworkNodeInfo>();
            _psoRingManager = new PsoRingManager();
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

        public void StartP2PAlgorithm()
        {
            TimerCallback timerCallback = RunP2pAlgorithm;
            _timer = new Timer(timerCallback, null,Miliseconds,Timeout.Infinite);
        }

        public void RunP2pAlgorithm(Object stateInfo)
        {
            Random random = new Random();
            switch (random.Next(0, 3))
            {
                case 0:
                    A1();
                    break;
                case 1:
                    A2();
                    break;
                case 2:
                    A5();
                    break;
            }

            _timer.Change(Miliseconds, Timeout.Infinite);
        }

        //Service part

        public void CloserPeerSearch(NetworkNodeInfo source)  //A3
        {
            if (NetworkNodeInfo.Distance(MyInfo,source) < _neighbors.Min(n => NetworkNodeInfo.Distance(MyInfo, n)))
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
       
        public void SuccessorCandidate(NetworkNodeInfo candidate)  //A4
        {
            _closerPeerSearchNodes.Add(candidate);
        }
        
        public void GetNeighbor(NetworkNodeInfo from, int j)  //A6
        {
            if (_neighbors.Count > j)
            {
                NodeServiceClient nodeServiceClient = new NodeServiceClient(from);
                nodeServiceClient.UpdateNeighbor(_neighbors[j], j);
            }
        }

        public void UpdateNeighbor(NetworkNodeInfo newNeighbor, int c) //A7
        {
            if (NetworkNodeInfo.Distance(_neighbors[c], newNeighbor) < NetworkNodeInfo.Distance(_neighbors[c], MyInfo))  //is it ok?
            {
                _neighbors[c + 1] = newNeighbor;
            }  //what if not?
        }

        //Client part
        public class NodeServiceClient : ClientBase<INodeService>, INodeService
        {
            public NodeServiceClient() { }

            public NodeServiceClient(string endpointConfigurationName) : base(endpointConfigurationName) { }

            public NodeServiceClient(string endpointConfigurationName, string remoteAddress) : base(endpointConfigurationName, remoteAddress) { }
            
            public NodeServiceClient(string endpointConfigurationName, EndpointAddress remoteAddress) : base(endpointConfigurationName, remoteAddress) { }
            
            public NodeServiceClient(Binding binding, EndpointAddress remotAddress) : base(binding, remotAddress) { }
            
            public NodeServiceClient(NetworkNodeInfo networkNodeInfo) : this(new NetTcpBinding(), networkNodeInfo.Address) { }  //configuration for NetTcpBinding?

            public void CloserPeerSearch(NetworkNodeInfo source)
            {
                Channel.CloserPeerSearch(source);
            }

            public void SuccessorCandidate(NetworkNodeInfo candidate)
            {
                Channel.SuccessorCandidate(candidate);
            }

            public void GetNeighbor(NetworkNodeInfo from, int which)
            {
                Channel.GetNeighbor(from, which);
            }

            public void UpdateNeighbor(NetworkNodeInfo newNeighbor, int which)
            {
                Channel.UpdateNeighbor(newNeighbor,which);
            }
        }

        public void A1()
        {
            if(_s!=null) _peers.Add(_s);
            _peers.UnionWith(_closerPeerSearchNodes);
            _peers.UnionWith(_searchMonitorNodes);
            _peers.UnionWith(_neighbors);

            _neighbors.Insert(0, GetClosestNeighbor());
            _searchMonitorNodes.Clear();
            _closerPeerSearchNodes.Clear();
        }

        public void A2()
        {
            Random random = new Random();  //do klasy?
            int r = random.Next(0, _neighbors.Count > 0 ? BootstrappingPeers.Count + 1 : BootstrappingPeers.Count);
            _s = r < BootstrappingPeers.Count ? BootstrappingPeers.ElementAt(r) : _neighbors[0];

            NodeServiceClient nodeServiceClient = new NodeServiceClient(_s);
            nodeServiceClient.CloserPeerSearch(MyInfo);
        }

        public void A5()
        {            
            for(int i=0; i<_neighbors.Count;++i)
            {
                NodeServiceClient nodeServiceClient = new NodeServiceClient(_neighbors[i]);
                nodeServiceClient.GetNeighbor(MyInfo, i);
            }
        }
    }
}
