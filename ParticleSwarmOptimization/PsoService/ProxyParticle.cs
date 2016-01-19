
using System;
using System.Linq;
using System.ServiceModel;
using Common;

namespace PsoService
{
    public class ProxyParticle
    {
        private static int _counter = 1;

        private IParticleService _particleClient;
        private IParticleService _particleService;
        private ServiceHost _host;
        public int Id { get; private set; }

        public Uri Address
        {
            get { return _host.BaseAddresses.First(); }
        }

        public Uri RemoteAddress { get; private set; }

        private ProxyParticle(string remoteNeighborAddress)
        {
            _particleClient = ParticleServiceClient.CreateClient(remoteNeighborAddress);
            RemoteAddress = new Uri(remoteNeighborAddress);
            Id = _counter++;
        }

        private ProxyParticle()
        {
            Id = _counter++;
        }

        public static ProxyParticle CreateProxyParticle(string remoteAddress, int nodeId)
        {
            var particle = new ProxyParticle(remoteAddress) { _particleService = new ParticleService() };
            particle._host = new ServiceHost(particle._particleService, new Uri(string.Format("net.tcp://0.0.0.0:{0}/{1}/particle/{2}", PortFinder.FreeTcpPort(), nodeId, particle.Id)));
            return particle;
        }
        public static ProxyParticle CreateProxyParticle(ulong nodeId)
        {
            var particle = new ProxyParticle() { _particleService = new ParticleService() };
            particle._host = new ServiceHost(particle._particleService, new Uri(string.Format("net.tcp://0.0.0.0:{0}/{1}/particle/{2}", PortFinder.FreeTcpPort(), nodeId, particle.Id)));
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
                return ParticleState.WorstState(1);
            }
            var s = _particleClient.GetBestState();
            _particleService.UpdateBestState(s);
            return _particleService.GetBestState();
        }

        public ParticleState GetBestState()
        {
            return _particleService.GetBestState();
        }
        public void UpdateBestState(ParticleState state)
        {
            _particleService.UpdateBestState(state);
        }

        public void UpdateRemoteAddress(Uri address)
        {
            RemoteAddress = address;
            _particleClient = ParticleServiceClient.CreateClient(address.ToString());
        }
    }
}