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
        NetworkNodeInfo[] Register(NetworkNodeInfo source);

        [OperationContract]
        void Deregister(NetworkNodeInfo brokenNodeInfo);

        [OperationContract]
        void StartCalculation(PsoParameters parameters);

        [OperationContract]
        void CalculationsFinished(NetworkNodeInfo source, object result);

        [OperationContract]
        void CheckStatus();
    }
}