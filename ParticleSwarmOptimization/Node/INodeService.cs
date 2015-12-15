using System.ServiceModel;

namespace Node
{
    [ServiceContract]
    public interface INodeService
    {
        [OperationContract]
        void CloserPeerSearch(NodeInfo source);

        [OperationContract]
        void SuccessorCandidate(NodeInfo candidate);

        [OperationContract]
        void GetNeighbor(NodeInfo from, int which);

        [OperationContract]
        void UpdateNeighbor(NodeInfo newNeighbor, int which);
    }
}
