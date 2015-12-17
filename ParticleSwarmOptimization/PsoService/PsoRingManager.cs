using System;
using System.Collections.Generic;
using System.Linq;
using Common;

namespace PsoService
{
    public class PsoRingManager : IPsoManager
    {
        private Tuple<NetworkNodeInfo, ProxyParticleService> _left;
        private Tuple<NetworkNodeInfo, ProxyParticleService> _right;
        public PsoSettings PsoSettings;

        public PsoRingManager(int nodeId)
        {
            _left = new Tuple<NetworkNodeInfo, ProxyParticleService>(null,ProxyParticleService.CreateProxyParticle(nodeId));
            _right = new Tuple<NetworkNodeInfo, ProxyParticleService>(null,ProxyParticleService.CreateProxyParticle(nodeId));
        }
        public void UpdatePsoNeighborhood(Tuple<NetworkNodeInfo, Uri[]>[] allNetworkNodes,
            NetworkNodeInfo currentNetworkNode)
        {
            //do nothing if there is only current network in collection
            if (allNetworkNodes.Length <= 1)
                return;
            var nodes = allNetworkNodes.OrderBy(t => t.Item1.Id).ToArray();
            Tuple<NetworkNodeInfo, Uri[]> previous = null, next = null;
            for (int index = 0; index < nodes.Length; index++)
            {
                var node = nodes[index];
                if (node.Item1.Id != currentNetworkNode.Id) continue;

                previous = nodes[(index - 1 + nodes.Length) % nodes.Length];
                next = nodes[(index + 1) % nodes.Length];
                break;
            }
            if (previous == null || next == null)
            {
                throw new ArgumentException("allNetworkNodes should include this node itself");
            }
            if (_left.Item1 ==null || previous.Item1.Id != _left.Item1.Id || !previous.Item2.Contains(_left.Item2.RemoteAddress))
            {
                _left.Item2.UpdateRemoteAddress(previous.Item2[0]);
            }
            if (_right.Item1 == null || next.Item1.Id != _right.Item1.Id || !next.Item2.Contains(_right.Item2.RemoteAddress))
            {
                _right.Item2.UpdateRemoteAddress(next.Item2[next.Item2.Length - 1]);
            }
        }

        public Uri[] GetProxyParticlesAddresses()
        {
            var uris = new List<Uri>();
            if (_left != null) uris.Add(_left.Item2.Address);
            if (_right != null) uris.Add(_right.Item2.Address);
            return uris.ToArray();
        }

        public PsoSettings CurrentProblem()
        {
            return PsoSettings;
        }

        public ParticleState Run(FitnessFunction fitnessFunction, PsoSettings psoSettings)
        {
            throw new NotImplementedException();
        }
    }
}