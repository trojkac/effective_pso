using System.ServiceModel;
using System.ServiceModel.Channels;

namespace Node
{
    public class NodeServiceClient : ClientBase<INodeService>, INodeService
    {
        public NodeServiceClient()
        {
        }

        public NodeServiceClient(string endpointConfigurationName)
            : base(endpointConfigurationName)
        {
        }

        public NodeServiceClient(string endpointConfigurationName, string remoteAddress)
            : base(endpointConfigurationName, remoteAddress)
        {
        }

        public NodeServiceClient(string endpointConfigurationName, EndpointAddress remoteAddress)
            : base(endpointConfigurationName, remoteAddress)
        {
        }

        public NodeServiceClient(Binding binding, EndpointAddress remotAddress)
            : base(binding, remotAddress)
        {
        }

        public NodeServiceClient(NetworkNodeInfo networkNodeInfo)
            : this(new NetTcpBinding(), networkNodeInfo.Address)
        {
        } //configuration for NetTcpBinding?

        public void CloserPeerSearch(NetworkNodeInfo source)
        {
            Channel.CloserPeerSearch(source);
        }

        public void SuccessorCandidate(NetworkNodeInfo candidate)
        {
            Channel.SuccessorCandidate(candidate);
        }

        public void GetNeighbor(NetworkNodeInfo from, int which)
        {
            Channel.GetNeighbor(from, which);
        }

        public void UpdateNeighbor(NetworkNodeInfo newNeighbor, int which)
        {
            Channel.UpdateNeighbor(newNeighbor, which);
        }
    }
}