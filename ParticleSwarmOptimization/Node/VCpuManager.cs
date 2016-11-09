using System;
using System.Collections.Generic;
using Common;
using Common.Parameters;
using Controller;
using NetworkManager;
using PsoService;
using System.Threading;

namespace Node
{
    public class VCpuManager
    {

        private NodeParameters _nodeParams;
        private NetworkNodeInfo _mainNodeInfo = null;
        public IPsoManager PsoRingManager { get; set; }
        private IPsoController _psoController { get; set; }



        public VCpuManager(string tcpAddress, int tcpPort, string pipeName, IPsoController psoController = null, IPsoManager psoRingManager = null)
        {
            NetworkNodeManager = new NetworkNodeManager(tcpAddress, tcpPort, pipeName);
            _psoController = psoController ?? new PsoController(NetworkNodeManager.NodeService.Info.Id);
            PsoRingManager = psoRingManager ?? new PsoRingManager(NetworkNodeManager.NodeService.Info.Id);
            NetworkNodeManager.NodeService.Info.ProxyParticlesAddresses = PsoRingManager.GetProxyParticlesAddresses();
            var uris = new List<Uri>();
            foreach (var address in NetworkNodeManager.NodeService.Info.ProxyParticlesAddresses)
            {
                var str = address.AbsoluteUri.Replace("0.0.0.0", tcpAddress);
                uris.Add(new Uri(str));
            }
            NetworkNodeManager.NodeService.Info.ProxyParticlesAddresses = uris.ToArray();

            NetworkNodeManager.NodeService.NeighborhoodChanged += PsoRingManager.UpdatePsoNeighborhood;
            NetworkNodeManager.NodeService.RegisterNode += RunOnNode;
            NetworkNodeManager.NodeService.StartCalculations += Run;
            NetworkNodeManager.NodeService.StopCalculations += () => _psoController.Stop();
            PsoRingManager.CommunicationLost += NetworkNodeManager.NodeService.Deregister;
            _psoController.CalculationsCompleted += Finish;
            NetworkNodeManager.NodeService.RemoteCalculationsFinished += _psoController.RemoteControllerFinished;

        }

        private void Finish(ParticleState result)
        {
            if (_mainNodeInfo == NetworkNodeManager.NodeService.Info)
            {
                var bestFromNodes = NetworkNodeManager.StopCalculations();
                _psoController.UpdateResultWithOtherNodes(bestFromNodes);
            }
            else
            {
                NetworkNodeManager.BroadcastCalculationsFinished(result);    
            }
            
        }


        public ParticleState GetResult()
        {
            _psoController.RunningAlgorithm.Wait();
            return _psoController.RunningAlgorithm.Result;
        }

        // NETWORK PART

        public NetworkNodeManager NetworkNodeManager { get; set; }
        public void StartTcpNodeService()
        {
            NetworkNodeManager.StartTcpNodeService();
        }
        public void CloseTcpNodeService()
        {
            NetworkNodeManager.CloseTcpNodeService();
        }
        public void StartPipeNodeService()
        {
            NetworkNodeManager.StartPipeNodeService();
        }
        public void ClosePipeNodeService()
        {
            NetworkNodeManager.ClosePipeNodeService();
        }


        public NetworkNodeInfo GetMyNetworkNodeInfo()
        {
            return NetworkNodeManager.NodeService.Info;
        }
        
        public void Run(PsoParameters psoParameters, NetworkNodeInfo mainNodeInfo)
        {
            _mainNodeInfo = mainNodeInfo;
            if (!_psoController.CalculationsRunning)
            {
                _psoController.Run(psoParameters, PsoRingManager.GetProxyParticles());
            }
        }

        public void StartCalculations(PsoParameters parameters, PsoParameters parametersToSend = null)
        {
            if (_psoController.CalculationsRunning) return;
            NetworkNodeManager.StartCalculations(parametersToSend ?? parameters);
            Run(parameters, NetworkNodeManager.NodeService.Info);
        }

        private void RunOnNode(NetworkNodeInfo newNode)
        {
            if (_psoController.CalculationsRunning)
            {
                NetworkNodeManager.StartCalculations(_psoController.RunningParameters, newNode);
            }
        }
    }
}