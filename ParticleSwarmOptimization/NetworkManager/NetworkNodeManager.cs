using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.ServiceModel;
using System.ServiceModel.Description;
using Common;

namespace NetworkManager
{
    public class NetworkNodeManager
    {

        private readonly string _pipeName = "NodeServicePipe";

        private readonly int _tcpPort = 8080;
        private ServiceHost _pipeHost;

        private ServiceHost _tcpHost;

        public NetworkNodeManager()
        {
            NodeService = new NodeService(_tcpPort, _pipeName);
        }

        public NetworkNodeManager(int tcpPort, string pipeName)
        {
            _tcpPort = tcpPort;
            _pipeName = pipeName;
            NodeService = new NodeService(_tcpPort, _pipeName);
        }


        public NodeService NodeService { get; set; }

        /// <summary>
        /// Returns list containing clients for all nodes in neighborhood except this node.
        /// </summary>
        public List<NodeServiceClient> NodeServiceClients
        {
            get
            {
                return NodeService
                    .KnownNodes
                    .Where(node => node.Id != NodeService.Info.Id)
                    .Select(neighbor => new TcpNodeServiceClient(neighbor))
                    .Cast<NodeServiceClient>()
                    .ToList();
            }
        }

        public void Register(NetworkNodeInfo info)
        {
            var client = new TcpNodeServiceClient(info);
            client.Register(NodeService.Info);
        }

        public void StartCalculations(PsoSettings settings)
        {
            foreach (var client in NodeServiceClients)
            {
                client.StartCalculation(settings);
            }
        }

        public void StartTcpNodeService()
        {
            var serviceAddress = "net.tcp://127.0.0.1:" + _tcpPort + "/NodeService";
            var serviceUri = new Uri(serviceAddress);

            _tcpHost = new ServiceHost(NodeService, serviceUri);

            _tcpHost.AddServiceEndpoint(typeof (INodeService), new NetTcpBinding(), "");

            var smb = new ServiceMetadataBehavior();
            _tcpHost.Description.Behaviors.Add(smb);
            var mexBinding = MetadataExchangeBindings.CreateMexTcpBinding();

            _tcpHost.AddServiceEndpoint(typeof (IMetadataExchange), mexBinding, "mex");

            try
            {
                _tcpHost.Open();
            }
            catch (CommunicationException ce)
            {
                _tcpHost.Abort();
            }
        }

        public void StartPipeNodeService()
        {
            var serviceAddress = "net.pipe://localhost/NodeService/" + _pipeName;
            var serviceUri = new Uri(serviceAddress);

            _pipeHost = new ServiceHost(NodeService, serviceUri);

            _pipeHost.AddServiceEndpoint(typeof (INodeService), new NetNamedPipeBinding(), "");

            var smb = new ServiceMetadataBehavior();
            _pipeHost.Description.Behaviors.Add(smb);
            var mexBinding = MetadataExchangeBindings.CreateMexNamedPipeBinding();

            _pipeHost.AddServiceEndpoint(typeof (IMetadataExchange), mexBinding, "mex");

            try
            {
                _pipeHost.Open();
            }
            catch (CommunicationException ce)
            {
                _pipeHost.Abort();
            }
        }

        public void CloseTcpNodeService()
        {
            Debug.WriteLine("Zamykam _tcpHost");
            _tcpHost.Close();
        }

        public void ClosePipeNodeService()
        {
            Debug.WriteLine("Zamykam _pipeHost");
            _pipeHost.Close();
        }
    }
}