using System;
using System.Linq;
using Common;

namespace Node
{
    public class PsoRingManager : IPsoManager
    {
        private Tuple<NetworkNodeInfo, ProxyParticleService> _left;
        private Tuple<NetworkNodeInfo, ProxyParticleService> _right;

        public PsoRingManager()
        {
        }
		

        /// <summary>
        /// Updates PSO neighborhood depending on the nodes network topology.
        /// PsoRingManager creates ring topology for PSO where left neighbor
        ///  to the current node is its predecessor and right neighbor is its 
        /// successor. For details of nodes topology see
        ///  https://ccl.northwestern.edu/papers/2005/ShakerReevesP2P.pdf
        /// </summary>
        /// <param name="allNetworkNodes">All nodes </param>
        /// <param name="currentNetworkNode">Current node info</param>
        public void UpdatePsoNeighborhood(NetworkNodeInfo[] allNetworkNodes, NetworkNodeInfo currentNetworkNode)
        {

            //TODO: Finish ring creation, check if left or right changed, update left and right
            var directNeighbours = allNetworkNodes.OrderBy(node => node.Distance(currentNetworkNode)).Take(2).ToArray();
            if (directNeighbours[0] - currentNetworkNode < 0)
            {

            }
            else
            {

            }
        }
    }
}