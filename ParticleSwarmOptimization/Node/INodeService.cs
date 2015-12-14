using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.ServiceModel;
using System.Text;

namespace Node
{
    // NOTE: You can use the "Rename" command on the "Refactor" menu to change the interface name "INodeService" in both code and config file together.
    [ServiceContract]
    public interface INodeService
    {
        [OperationContract]
        void CloserNodeSearch(NodeInfo source);

        [OperationContract]
        void SuccessorCandidate(NodeInfo candidate);

        [OperationContract]
        void GetNeighbor(int which);

        [OperationContract]
        void UpdateNeighbor(NodeInfo newNeighbor, int which);

        [OperationContract]
        int GetMinAvailableId();

    }
}
