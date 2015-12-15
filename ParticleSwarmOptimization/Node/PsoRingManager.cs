using System;
using System.Linq;
using Common;

namespace Node
{
    public class PsoRingManager : IPsoManager, IPsoService
    {
        private Tuple<NodeInfo, ProxyParticle> _left;
        private Tuple<NodeInfo, ProxyParticle> _right;

        public PsoRingManager()
        {
        }
		
        /// <summary>
        /// Operation exposed for ProxyParticles of the other nodes
        /// </summary>
        /// <param name="nodeId">Source of request</param>
        /// <returns>Best known position of appropriate proxy particle</returns>
        public ParticleState GetBestState(int nodeId)
        {
            //Check wheter nodeId is left or right and return appropriate ProxyParticle::GetLocalBest()
            // If it is neither left nor right nodeId we have to decide what to do:
            //  1) respond with an error, 
            //  2) respond with a fitness value which won't be considered (Infinity for minimalization),
            //=>3) respond with any of left/right best known solution
            return _left != null && _left.Item1.Id == nodeId ? 
                _left.Item2.GetLocalBest()  : 
                _right.Item2.GetLocalBest() ?? new ParticleState(new double[]{1.0},double.PositiveInfinity ) ;

        }

        /// <summary>
        /// Updates PSO neighborhood depending on the nodes network topology.
        /// PsoRingManager creates ring topology for PSO where left neighbor
        ///  to the current node is its predecessor and right neighbor is its 
        /// successor. For details of nodes topology see
        ///  https://ccl.northwestern.edu/papers/2005/ShakerReevesP2P.pdf
        /// </summary>
        /// <param name="allNodes">All nodes </param>
        /// <param name="currentNode">Current node info</param>
        public void UpdatePsoNeighborhood(NodeInfo[] allNodes, NodeInfo currentNode)
        {

            //TODO: Finish ring creation, check if left or right changed, update left and right
            var directNeighbours = allNodes.OrderBy(node => node.Distance(currentNode)).Take(2).ToArray();
            if (directNeighbours[0] - currentNode < 0)
            {

            }
            else
            {

            }
        }
    }
}