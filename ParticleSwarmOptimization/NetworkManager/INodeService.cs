using System;
using System.ServiceModel;
using Common;
using Common.Parameters;

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
        void StartCalculation(PsoParameters parameters, NetworkNodeInfo mainNodeInfo);

        [OperationContract]
        ParticleState StopCalculation();

        [OperationContract]
        void CalculationsFinished(NetworkNodeInfo source, ParticleState result);

        [OperationContract]
        void CheckStatus();
    }
}