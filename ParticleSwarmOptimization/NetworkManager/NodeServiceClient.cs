using System;
using System.ServiceModel;
using System.ServiceModel.Channels;
using Common;

namespace NetworkManager
{
    public abstract class NodeServiceClient : INodeService
    {
        public EndpointAddress Address;
        public Binding Binding;
        public IChannelFactory<INodeService> ChannelFactory;
        public INodeService Proxy;


        public void UpdateNodes(NetworkNodeInfo[] nodes)
        {
            Proxy.UpdateNodes(nodes);
        }

        public NetworkNodeInfo[] Register(NetworkNodeInfo source)
        {
            return Proxy.Register(source);
        }

        public void Deregister(NetworkNodeInfo brokenNodeInfo)
        {
            Proxy.Deregister(brokenNodeInfo);
        }


        public void StartCalculation(PsoSettings settings)
        {
            Proxy.StartCalculation(settings);
        }

        public void CheckStatus()
        {
            Proxy.CheckStatus();
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