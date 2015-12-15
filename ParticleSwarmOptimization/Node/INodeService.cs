using System.ServiceModel;

namespace Node
{
    [ServiceContract]
    public interface INodeService
    {
        [OperationContract]
        void CloserPeerSearch(NetworkNodeInfo source);

        [OperationContract]
        void SuccessorCandidate(NetworkNodeInfo candidate);

        [OperationContract]
        void GetNeighbor(NetworkNodeInfo from, int which);

        [OperationContract]
        void UpdateNeighbor(NetworkNodeInfo newNeighbor, int which);
    }
}
