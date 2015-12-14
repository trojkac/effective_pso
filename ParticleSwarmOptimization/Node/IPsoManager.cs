namespace Node
{
    public interface IPsoManager
    {
        /// <summary>
        /// Class implementing IPsoManager uses UpdatePsoNeighborhood to create connections between PSO swarms independant from cluster's infrastructure.
        /// </summary>
        /// <param name="allNodes"> All nodes in the cluster</param>
        /// <param name="currentNode"> Current's node info</param>
        void UpdatePsoNeighborhood(NodeInfo[] allNodes, NodeInfo currentNode);
    }
}