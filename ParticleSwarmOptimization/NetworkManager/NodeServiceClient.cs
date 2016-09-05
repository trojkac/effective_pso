using System;
using System.ServiceModel;
using System.ServiceModel.Channels;
using Common;
using Common.Parameters;

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
            try
            {
                Proxy.UpdateNodes(nodes);
            }
            catch
            {

            }
        }

        public NetworkNodeInfo[] Register(NetworkNodeInfo source)
        {
            try
            {
                return Proxy.Register(source);

            }
            catch
            {

            }
            return null;
        }

        public void Deregister(NetworkNodeInfo brokenNodeInfo)
        {
            try
            {
                Proxy.Deregister(brokenNodeInfo);

            }
            catch
            {

            }
        }


        public void StartCalculation(PsoParameters parameters, NetworkNodeInfo mainNodeInfo)
        {
            try
            {
                Proxy.StartCalculation(parameters, mainNodeInfo);

            }
            catch
            {

            }
        }

        public ParticleState StopCalculation()
        {
            try
            {
                return Proxy.StopCalculation();

            }
            catch
            {

            }
            return new ParticleState();
        }

        public void CalculationsFinished(NetworkNodeInfo source, ParticleState result)
        {
            try
            {
                Proxy.CalculationsFinished(source, result);

            }
            catch
            {
                // ignored
            }
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
            Binding = new NetTcpBinding(SecurityMode.None);
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