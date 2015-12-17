using System;
using System.Diagnostics;
using System.ServiceModel;
using System.ServiceModel.Channels;
using System.ServiceModel.Description;
using System.Threading;

namespace Node
{
    public class NodeManager
    {
        private const int Miliseconds = 1000;
        private Timer _timer;

        public NodeService NodeService { get; set; }
        private ServiceHost _serviceHost;

        public NodeManager(NodeService nodeService)
        {
            NodeService = nodeService;
        }

        public void StartNodeService()
        {
            _serviceHost = new ServiceHost(NodeService, NodeService.MyInfo.Address.Uri);

            try
            {
                //adding service endpoint
                Binding binding = new NetTcpBinding();
                String relativeAddress = "NodeService";
                _serviceHost.AddServiceEndpoint(typeof(INodeService), binding, relativeAddress);

                //starting service
                Debug.WriteLine("Otwieram serwer pod adresem: " + NodeService.MyInfo.Address + relativeAddress);
                _serviceHost.Open();

                TimerCallback timerCallback = RunP2PAlgorithm;
                _timer = new Timer(timerCallback, null, Miliseconds, Timeout.Infinite);
            }
            catch (CommunicationException ce)
            {
                _serviceHost.Abort();
            }
        }

        public void RunP2PAlgorithm(Object stateInfo)
        {
            Random random = new Random();
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

        public void CloseNodeService()
        {
            Debug.WriteLine("Zamykam serwer pod adresem: " + NodeService.MyInfo.Address);
            _serviceHost.Close();
        }
    }
}