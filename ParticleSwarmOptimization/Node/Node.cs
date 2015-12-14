using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.ServiceModel;
using System.ServiceModel.Description;
using System.Text;

namespace Node
{
    [ServiceBehavior(InstanceContextMode = InstanceContextMode.Single)]
    public class Node : INodeService
    {
        private readonly List<NodeInfo> _bootstrap;
        private readonly NodeInfo _myInfo;

        private List<NodeInfo> _peers;
        private readonly List<NodeInfo> _searchMonitorNodes;
        private List<NodeInfo> _closerPeerSearchNodes;

        private List<NodeInfo> _neighborhood;




        public Node(List<NodeInfo> bootstrap, NodeInfo myInfo)
        {
            _bootstrap = bootstrap;
            _myInfo = myInfo;
            _peers = new List<NodeInfo>();
            _searchMonitorNodes = new List<NodeInfo>();
            _closerPeerSearchNodes = new List<NodeInfo>();
            _neighborhood = new List<NodeInfo>();
        }


        public void CloserNodeSearch(NodeInfo source)
        {
            if (source.Distance(_myInfo) < _neighborhood.Min(n => n.Distance(_myInfo)))
            {
                _searchMonitorNodes.Add(source);
            }
            throw new NotImplementedException();
        }

        public void SuccessorCandidate(NodeInfo candidate)
        {
            throw new NotImplementedException();
        }

        public void GetNeighbor(int which)
        {
            throw new NotImplementedException();
        }

        public void UpdateNeighbor(NodeInfo newNeighbor, int which)
        {
            throw new NotImplementedException();
        }

        public int GetMinAvailableId()
        {
            throw new NotImplementedException();
        }
    }
}
