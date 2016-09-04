using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.ServiceModel;
using System.ServiceModel.Description;
using System.Threading;
using System.Threading.Tasks;
using Common;
using Common.Parameters;

namespace NetworkManager
{
    public class NetworkNodeManager
    {

        private readonly string _pipeName = "NodeServicePipe";

        
        private readonly int _tcpPort = 8080;
        private ServiceHost _pipeHost;
        private ServiceHost _tcpHost;
        private Timer _statusTimer;

        public NetworkNodeManager(string tcpAddress)
        {
            NodeService = new NodeService(tcpAddress, _tcpPort, _pipeName);
        }

        public NetworkNodeManager(string tcpAddress,int tcpPort, string pipeName)
        {
            _tcpPort = tcpPort;
            _pipeName = pipeName;
            NodeService = new NodeService(tcpAddress, _tcpPort, _pipeName);
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
            var neighbors = client.Register(NodeService.Info);
            NodeService.UpdateNodes(neighbors);

        }

        public void StartCalculations(PsoParameters parameters)
        {
            foreach (var client in NodeServiceClients)
            {
                try
                {
                    client.StartCalculation(parameters);
                }
                catch
                {
                    Debug.WriteLine("cannot start calculations on {0}",parameters);
                }
                
            }
        }

        public void StartCalculations(PsoParameters parameters, NetworkNodeInfo target)
        {
            var client = new TcpNodeServiceClient(target);
            try
            {
                client.StartCalculation(parameters);
            }
            catch
            {
                Debug.WriteLine("cannot start calculations on {0}", parameters);
            }
        }

        public void FinishCalculations(ParticleState result)
        {
            foreach (var client in NodeServiceClients)
            {
                client.CalculationsFinished(NodeService.Info,result);
            }
        }

        public void StartTcpNodeService()
        {
            var serviceAddress = "net.tcp://0.0.0.0:" + _tcpPort + "/NodeService";
            var serviceUri = new Uri(serviceAddress);

            _tcpHost = new ServiceHost(NodeService, serviceUri);

            _tcpHost.AddServiceEndpoint(typeof (INodeService), new NetTcpBinding(SecurityMode.None), "");

            var smb = new ServiceMetadataBehavior();
            _tcpHost.Description.Behaviors.Add(smb);
            var mexBinding = MetadataExchangeBindings.CreateMexTcpBinding();

            _tcpHost.AddServiceEndpoint(typeof (IMetadataExchange), mexBinding, "mex");

            try
            {
                _tcpHost.Open();
               // _statusTimer = new Timer(CheckStatuses, null, 5000, 10000);
            }
            catch (CommunicationException ce)
            {
               // _statusTimer.Dispose();
                _tcpHost.Abort();
                
            }
        }

        public void StartPipeNodeService()
        {
            throw new NotImplementedException();
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
            throw new NotImplementedException();
            Debug.WriteLine("Zamykam _pipeHost");
            _pipeHost.Close();
        }
    }
}