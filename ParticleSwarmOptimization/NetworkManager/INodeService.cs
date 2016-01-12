using System;
using System.ServiceModel;
using Common;

namespace NetworkManager
{
    [ServiceContract]
    public interface INodeService
    {
        [OperationContract] 
        void UpdateNodes(NetworkNodeInfo[] nodes);

        [OperationContract]
        void Register(NetworkNodeInfo source);

        [OperationContract]
        void StartCalculation(PsoSettings settings);
    }
}