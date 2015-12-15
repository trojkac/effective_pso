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

        private readonly NodeInfo _myInfo;
        private PsoRingManager _psoRingManager;

        //CONST
        private readonly HashSet<NodeInfo> _bootstrappingPeers;

        //INPUT
        //private NodeInfo _w;
        //private NodeInfo _x;
        //private int _c;
        //private NodeInfo _z;
        private NodeInfo _s;

        //VAR
        private readonly HashSet<NodeInfo> _peers;  //S
        private readonly HashSet<NodeInfo> _searchMonitorNodes;  //B
        private readonly HashSet<NodeInfo> _closerPeerSearchNodes;  //W
        private readonly List<NodeInfo> _neighbors;  //Gamma   //czy nie trzeba uważać na przypadek, gdy to jest puste?


        public Node(HashSet<NodeInfo> bootstrap, NodeInfo myInfo)
        {
            _bootstrappingPeers = bootstrap;
            _myInfo = myInfo;
            _peers = new HashSet<NodeInfo>();
            _searchMonitorNodes = new HashSet<NodeInfo>();
            _closerPeerSearchNodes = new HashSet<NodeInfo>();
            _neighbors = new List<NodeInfo>();
            _psoRingManager = new PsoRingManager();
            StartP2PAlgorithm();
        }

        public NodeInfo GetClosestNeighbor()
        {
            int minDistance = Int32.MaxValue;
            NodeInfo closestNeighbor = null;
            foreach (NodeInfo nodeInfo in _peers)
            {
                if (NodeInfo.Distance(_myInfo, nodeInfo) < minDistance)
                {
                    minDistance = NodeInfo.Distance(_myInfo, nodeInfo);
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

        public void CloserPeerSearch(NodeInfo source)  //A3
        {
            if (NodeInfo.Distance(_myInfo,source) < _neighbors.Min(n => NodeInfo.Distance(_myInfo, n)))
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
       
        public void SuccessorCandidate(NodeInfo candidate)  //A4
        {
            _closerPeerSearchNodes.Add(candidate);
        }
        
        public void GetNeighbor(NodeInfo from, int j)  //A6
        {
            if (_neighbors.Count > j)
            {
                NodeServiceClient nodeServiceClient = new NodeServiceClient(from);
                nodeServiceClient.UpdateNeighbor(_neighbors[j], j);
            }
        }

        public void UpdateNeighbor(NodeInfo newNeighbor, int c) //A7
        {
            if (NodeInfo.Distance(_neighbors[c], newNeighbor) < NodeInfo.Distance(_neighbors[c], _myInfo))  //is it ok?
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
            
            public NodeServiceClient(NodeInfo nodeInfo) : this(new NetTcpBinding(), nodeInfo.Address) { }  //configuration for NetTcpBinding?

            public void CloserPeerSearch(NodeInfo source)
            {
                Channel.CloserPeerSearch(source);
            }

            public void SuccessorCandidate(NodeInfo candidate)
            {
                Channel.SuccessorCandidate(candidate);
            }

            public void GetNeighbor(NodeInfo from, int which)
            {
                Channel.GetNeighbor(from, which);
            }

            public void UpdateNeighbor(NodeInfo newNeighbor, int which)
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
            int r = random.Next(0, _neighbors.Count > 0 ? _bootstrappingPeers.Count + 1 : _bootstrappingPeers.Count);
            _s = r < _bootstrappingPeers.Count ? _bootstrappingPeers.ElementAt(r) : _neighbors[0];

            NodeServiceClient nodeServiceClient = new NodeServiceClient(_s);
            nodeServiceClient.CloserPeerSearch(_myInfo);
        }

        public void A5()
        {            
            for(int i=0; i<_neighbors.Count;++i)
            {
                NodeServiceClient nodeServiceClient = new NodeServiceClient(_neighbors[i]);
                nodeServiceClient.GetNeighbor(_myInfo, i);
            }
        }
    }
}
