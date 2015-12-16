using System;
using System.Diagnostics;
using System.Net;
using System.ServiceModel;
using System.ServiceModel.Description;
using Common;

namespace Node
{
    [ServiceBehavior(InstanceContextMode = InstanceContextMode.Single)]
    public class ProxyParticle : IPsoService
    {
        private static int _counter = 1;
       
        private IPsoService _psoClient;
        private ParticleState _bestKnownState;
        private ServiceHost _host;
        public int Id { get; private set; }

        private ProxyParticle(string remoteNeighborAddress)
        {
            _bestKnownState = new ParticleState(new[] { 0.0 }, double.PositiveInfinity);
            _psoClient = new PsoServiceClient("particleProxyClientTcp", remoteNeighborAddress);
            Id = _counter++;
        }

        public static ProxyParticle CreateProxyParticle(string remoteAddress,int nodeId)
        {
            var particle = new ProxyParticle(remoteAddress);
            particle._host = new ServiceHost(particle, new Uri(String.Format("net.tcp://localhost/{0}/particle/",nodeId)));
            return particle;
        }

        public void Open()
        {
            _host.Open();
        }

        public void Close()
        {
            _host.Close();
        }

        public void UpdateRemoteAddress(string address)
        {
            _psoClient = new PsoServiceClient("particleProxyClientTcp", address);
        }

        /// <summary>
        /// Function called by the other particle in local swarm to know this particle's personal best
        /// which is personal best of the linked particle in the other swarm
        /// </summary>
        /// <returns></returns>
        public ParticleState GetRemoteBest()
        {
            var s = _psoClient.GetBestState();
            if (s.FitnessValue < _bestKnownState.FitnessValue)
            {
                _bestKnownState = s;
            }
            return _bestKnownState;
        }


        public ParticleState GetBestState()
        {
            return _bestKnownState;
        }
    }
}