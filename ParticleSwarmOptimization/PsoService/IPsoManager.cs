using System;
using Common;

namespace PsoService
{
    public interface IPsoManager
    {
        /// <summary>
        /// Class implementing IPsoManager uses UpdatePsoNeighborhood to create connections between PSO swarms independant from cluster's infrastructure.
        /// </summary>
        /// <param name="allNetworkNodes"> All nodes in the cluster with collection of URIs of proxy particles for every node </param>
        /// <param name="currentNetworkNode"> Current's node info</param>
        void UpdatePsoNeighborhood(NetworkNodeInfo[] allNetworkNodes, NetworkNodeInfo currentNetworkNode);

        /// <summary>
        /// Returns ProxyParticle endpoints URIs to be used by other particles
        /// </summary>
        /// <returns>array of base addresses of IProxyService used in this PsoManager</returns>
        Uri[] GetProxyParticlesAddresses();

        ProxyParticle[] GetProxyParticles();
        event CommunicationBreakdown CommunicationLost;
    }
}
