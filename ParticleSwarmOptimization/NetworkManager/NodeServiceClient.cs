using System;
using System.ServiceModel;
using System.ServiceModel.Channels;
using Common;

namespace NetworkManager
{
    public abstract class NodeServiceClient
    {
        public EndpointAddress Address;
        public Binding Binding;
        public IChannelFactory<INodeService> ChannelFactory;
        public INodeService Proxy;

        public void CloserPeerSearch(NetworkNodeInfo source)
        {
            Proxy.CloserPeerSearch(source);
        }

        public void SuccessorCandidate(NetworkNodeInfo candidate)
        {
            Proxy.SuccessorCandidate(candidate);
        }

        public void GetNeighbor(NetworkNodeInfo from, int which)
        {
            Proxy.GetNeighbor(from, which);
        }

        public void UpdateNeighbor(NetworkNodeInfo newNeighbor, int which)
        {
            Proxy.UpdateNeighbor(newNeighbor, which);
        }

        public Object ReceiveMessage(Object msg, NetworkNodeInfo src, NetworkNodeInfo dst)
        {
            return Proxy.ReceiveMessage(msg, src, dst);
        }

        public Tuple<NetworkNodeInfo, Uri[]>[] GetProxyParticlesAddresses(NetworkNodeInfo src)
        {
            return Proxy.GetProxyParticlesAddresses(src);
        }
    }

    public class TcpNodeServiceClient : NodeServiceClient
    {
        public TcpNodeServiceClient(string tcpAddress)
        {
            Address = new EndpointAddress(tcpAddress);
            Binding = new NetTcpBinding();
            ChannelFactory = new ChannelFactory<INodeService>(Binding);
            Proxy = ChannelFactory.CreateChannel(Address);
        }

        public TcpNodeServiceClient(EndpointAddress tcpAddress)
            : this(tcpAddress.ToString())
        {
        }

        public TcpNodeServiceClient(NetworkNodeInfo nodeInfo)
            : this(nodeInfo.TcpAddress)
        {
        }
    }

    public class PipeNodeServiceClient : NodeServiceClient
    {
        public PipeNodeServiceClient(string pipeAddress)
        {
            Address = new EndpointAddress(pipeAddress);
            Binding = new NetNamedPipeBinding();
            ChannelFactory = new ChannelFactory<INodeService>(Binding);
            Proxy = ChannelFactory.CreateChannel(Address);
        }

        public PipeNodeServiceClient(EndpointAddress pipeAddress)
            : this(pipeAddress.ToString())
        {
        }

        public PipeNodeServiceClient(NetworkNodeInfo nodeInfo)
            : this(nodeInfo.PipeAddress)
        {
        }
    }
}