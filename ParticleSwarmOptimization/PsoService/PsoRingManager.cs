using System;
using System.Collections.Generic;
using System.Linq;
using Common;

namespace PsoService
{
    public delegate void CommunicationBreakdown(NetworkNodeInfo brokenNode);
    public class PsoRingManager : IPsoManager
    {
        public event CommunicationBreakdown CommunicationLost;
        private Tuple<NetworkNodeInfo, ProxyParticle> _left;
        private Tuple<NetworkNodeInfo, ProxyParticle> _right;
        public PsoRingManager(ulong nodeId)
        {
            _left = new Tuple<NetworkNodeInfo, ProxyParticle>(null, ProxyParticle.CreateProxyParticle(nodeId));
            _right = new Tuple<NetworkNodeInfo, ProxyParticle>(null, ProxyParticle.CreateProxyParticle(nodeId));

            _left.Item2.CommunicationBreakdown += OnLeftCommunicationFailure;
            _right.Item2.CommunicationBreakdown += OnRightCommunicationFailure;


        }

        void OnLeftCommunicationFailure()
        {
            if(CommunicationLost != null)
            CommunicationLost(_left.Item1);
        }
        void OnRightCommunicationFailure()
        {
            if (CommunicationLost != null)
                CommunicationLost(_right.Item1);
        }

        public void UpdatePsoNeighborhood(NetworkNodeInfo[] allNetworkNodes,
            NetworkNodeInfo currentNetworkNode)
        {
            //do nothing if there is only current network in collection
            if (allNetworkNodes == null || allNetworkNodes.Length < 1)
                return;
            var nodes = allNetworkNodes.OrderBy(t => t.Id).ToArray();
            NetworkNodeInfo previous = null, next = null;
            for (int index = 0; index < nodes.Length; index++)
            {
                var node = nodes[index];
                if (node.Id != currentNetworkNode.Id) continue;

                previous = nodes[(index - 1 + nodes.Length) % nodes.Length];
                next = nodes[(index + 1) % nodes.Length];
                break;
            }
            if (previous == null || next == null)
            {
                throw new ArgumentException("allNetworkNodes should include this node itself");
            }
            if (_left.Item1 == null || previous.Id != _left.Item1.Id || !previous.ProxyParticlesAddresses.Contains(_left.Item2.RemoteAddress))
            {
                _left.Item2.UpdateRemoteAddress(previous.ProxyParticlesAddresses[0]);
            }
            if (_right.Item1 == null || next.Id != _right.Item1.Id || !next.ProxyParticlesAddresses.Contains(_right.Item2.RemoteAddress))
            {
                _right.Item2.UpdateRemoteAddress(next.ProxyParticlesAddresses[next.ProxyParticlesAddresses.Length - 1]);
            }
        }

        public Uri[] GetProxyParticlesAddresses()
        {
            var uris = new List<Uri>();
            if (_left != null) uris.Add(_left.Item2.Address);
            if (_right != null) uris.Add(_right.Item2.Address);
            return uris.ToArray();
        }

        public ProxyParticle[] GetProxyParticles()
        {
            return new[] {_left.Item2, _right.Item2};
        }
    }
}