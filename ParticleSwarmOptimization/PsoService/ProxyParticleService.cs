
using System;
using System.Linq;
using System.Net.Sockets;
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

        public Uri Address
        {
            get { return _host.BaseAddresses.First(); }
        }

        public Uri RemoteAddress { get; private set; }


        private ProxyParticleService(string remoteNeighborAddress)
        {
            _bestKnownState = new ParticleState(new[] { 0.0 }, double.PositiveInfinity);
            _particleClient = new ParticleServiceClient("particleProxyClientTcp", remoteNeighborAddress);
            RemoteAddress = new Uri(remoteNeighborAddress);
            Id = _counter++;
        }

        private ProxyParticleService()
        {
            _bestKnownState = ParticleState.WorstState;
            Id = _counter++;
        }


        public static ProxyParticleService CreateProxyParticle(string remoteAddress, int nodeId)
        {
            var particle = new ProxyParticleService(remoteAddress);
            particle._host = new ServiceHost(particle, new Uri(string.Format("net.tcp://localhost:{0}/{1}/particle/{2}", PortFinder.FreeTcpPort(), nodeId, particle.Id)));
            return particle;
        }
        public static ProxyParticleService CreateProxyParticle(int nodeId)
        {
            var particle = new ProxyParticleService();
            particle._host = new ServiceHost(particle, new Uri(string.Format("net.tcp://localhost:{0}/{1}/particle/{2}", PortFinder.FreeTcpPort(), nodeId, particle.Id)));
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
            RemoteAddress = new Uri(address);

        }

        /// <summary>
        /// Function called by the other particle in local swarm to know this particle's personal best
        /// which is personal best of the linked particle in the other swarm
        /// </summary>
        /// <returns></returns>
        public ParticleState GetRemoteBest()
        {
            if (_particleClient == null)
            {
                return ParticleState.WorstState;
            }
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

        public void UpdateRemoteAddress(Uri address)
        {
            RemoteAddress = address;
            _particleClient = new ParticleServiceClient("particleProxyClientTcp", address.ToString());
        }
    }
}