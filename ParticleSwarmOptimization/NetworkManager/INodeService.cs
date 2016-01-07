using System;
using System.Collections.Generic;
using System.ServiceModel;

namespace NetworkManager
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
        void UpdateNeighbor(NetworkNodeInfo potentialNeighbor, int which);

        [OperationContract]
        Object ReceiveMessage(Object msg, NetworkNodeInfo src, NetworkNodeInfo dst);

        [OperationContract]
        Tuple<NetworkNodeInfo, Uri[]>[] GetProxyParticlesAddresses(NetworkNodeInfo src);
    }
}