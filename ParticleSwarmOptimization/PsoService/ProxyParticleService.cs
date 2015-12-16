
using System;
using System.ServiceModel;
using Common;

namespace PsoService
{
    [ServiceBehavior(InstanceContextMode = InstanceContextMode.Single)]
    public class ProxyParticleService : IParticleService
    {
        private static int _counter = 1;
       
        private IParticleService _particleClient;
        private ParticleState _bestKnownState;
        private ServiceHost _host;
        public int Id { get; private set; }

        private ProxyParticleService(string remoteNeighborAddress)
        {
            _bestKnownState = new ParticleState(new[] { 0.0 }, double.PositiveInfinity);
            _particleClient = new ParticleServiceClient("particleProxyClientTcp", remoteNeighborAddress);
            Id = _counter++;
        }

        public static ProxyParticleService CreateProxyParticle(string remoteAddress,int nodeId)
        {
            var particle = new ProxyParticleService(remoteAddress);
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
            _particleClient = new ParticleServiceClient("particleProxyClientTcp", address);
        }

        /// <summary>
        /// Function called by the other particle in local swarm to know this particle's personal best
        /// which is personal best of the linked particle in the other swarm
        /// </summary>
        /// <returns></returns>
        public ParticleState GetRemoteBest()
        {
            var s = _particleClient.GetBestState();
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
        public void UpdateBestState(ParticleState state)
        {
            _bestKnownState = state;
        }
        
    }
}