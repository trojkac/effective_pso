using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.ServiceModel;
using System.ServiceModel.Channels;
using System.ServiceModel.Description;
using System.Threading;

namespace NetworkManager
{
    public class NetworkNodeManager
    {
        private Random random = new Random();
        private const int Miliseconds = 4000;
        private Timer _timer;

        public NodeService NodeService { get; set; }

        public List<NodeServiceClient> NodeServiceClients
        {
            get
            {
                List<NetworkNodeInfo> neighbors = NodeService.GetNeighbors();
                List<NodeServiceClient> clients = new List<NodeServiceClient>();
                foreach (NetworkNodeInfo neighbor in neighbors)
                {
                    clients.Add(new TcpNodeServiceClient(neighbor));
                }

                return clients;
            }
        }

        private ServiceHost _tcpHost;
        private ServiceHost _pipeHost;

        private int _tcpPort = 8080;
        private string _pipeName = "NodeServicePipe";

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

        public void StartTcpNodeService()
        {
            string serviceAddress = "net.tcp://localhost:" + _tcpPort + "/NodeService";
            Uri serviceUri = new Uri(serviceAddress);

            _tcpHost = new ServiceHost(NodeService, serviceUri);

            _tcpHost.AddServiceEndpoint(typeof(INodeService), new NetTcpBinding(), "");

            ServiceMetadataBehavior smb = new ServiceMetadataBehavior();
            _tcpHost.Description.Behaviors.Add(smb);
            Binding mexBinding = MetadataExchangeBindings.CreateMexTcpBinding();

            _tcpHost.AddServiceEndpoint(typeof(IMetadataExchange), mexBinding, "mex");

            try
            {
                _tcpHost.Open();
                StartP2PTimer();
            }
            catch (CommunicationException ce)
            {
                _tcpHost.Abort();
            }
        }

        public void StartPipeNodeService()
        {
            string serviceAddress = "net.pipe://localhost/NodeService/" + _pipeName;
            Uri serviceUri = new Uri(serviceAddress);

            _pipeHost = new ServiceHost(NodeService, serviceUri);

            _pipeHost.AddServiceEndpoint(typeof(INodeService), new NetNamedPipeBinding(), "");

            ServiceMetadataBehavior smb = new ServiceMetadataBehavior();
            _pipeHost.Description.Behaviors.Add(smb);
            Binding mexBinding = MetadataExchangeBindings.CreateMexNamedPipeBinding();

            _pipeHost.AddServiceEndpoint(typeof(IMetadataExchange), mexBinding, "mex");

            try
            {
                _pipeHost.Open();
                StartP2PTimer();
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

        public void StartP2PTimer()
        {
            TimerCallback timerCallback = RunP2PAlgorithm;
            _timer = new Timer(timerCallback, null, Miliseconds, Timeout.Infinite);
        }

        public void RunP2PAlgorithm(Object stateInfo)
        {          
            switch (random.Next(0, 3))
            {
                case 0:
                    NodeService.A1();
                    break;
                case 1:
                    NodeService.A2();
                    break;
                case 2:
                    NodeService.A5();
                    break;
            }

            _timer.Change(Miliseconds, Timeout.Infinite);
        }

        public void AddBootstrappingPeer(NetworkNodeInfo peerInfo)
        {
            NodeService.AddBootstrappingPeer(peerInfo);
        }

        public void SetBootstrappingPeers(HashSet<NetworkNodeInfo> peers)
        {
            NodeService.SetBootstrappingPeers(peers);
        }

        public List<Uri> GetAllProxyParticleAddresses()
        {
            List<Uri> addresses = new List<Uri>();
            return addresses;
        } 
    }
}
