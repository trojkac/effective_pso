namespace Node
{
    public interface IPsoManager
    {
        /// <summary>
        /// Class implementing IPsoManager uses UpdatePsoNeighborhood to create connections between PSO swarms independant from cluster's infrastructure.
        /// </summary>
        /// <param name="allNetworkNodes"> All nodes in the cluster</param>
        /// <param name="currentNetworkNode"> Current's node info</param>
        void UpdatePsoNeighborhood(NetworkNodeInfo[] allNetworkNodes, NetworkNodeInfo currentNetworkNode);
    }
}