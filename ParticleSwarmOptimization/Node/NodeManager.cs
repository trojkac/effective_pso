using System;
using System.ServiceModel;
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
            _serviceHost = new ServiceHost(typeof(INodeService), NodeService.MyInfo.Address.Uri);
            _serviceHost.Open();

            TimerCallback timerCallback = RunP2PAlgorithm;
            _timer = new Timer(timerCallback, null, Miliseconds, Timeout.Infinite);
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
            _serviceHost.Close();
        }
    }
}