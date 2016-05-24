
using System;
using System.Diagnostics;
using System.Linq;
using System.ServiceModel;
using Algorithm;
using Common;

namespace PsoService
{
    
    public delegate void ParticleCommunicationBreakdown();
    public class ProxyParticle : Particle
    {
        public event ParticleCommunicationBreakdown CommunicationBreakdown;
       
        private IParticleService _particleClient;
        private IParticleService _particleService;
        private ServiceHost _host;
        public NetworkNodeInfo RemoteNode{
            get{
                return new NetworkNodeInfo(RemoteAddress.Authority,"");
            }
        }
        public Uri Address
        {
            get { return _host.BaseAddresses.First(); }
        }

        public Uri RemoteAddress { get; private set; }

        private ProxyParticle(string remoteNeighborAddress)
        {
            _particleClient = ParticleServiceClient.CreateClient(remoteNeighborAddress);
            RemoteAddress = new Uri(remoteNeighborAddress);
        }

        private ProxyParticle()
        {
        }

        public static ProxyParticle CreateProxyParticle(ulong nodeId)
        {
            var particle = new ProxyParticle() { _particleService = new ParticleService() };
            particle._host = new ServiceHost(particle._particleService, new Uri(string.Format("net.tcp://0.0.0.0:{0}/{1}/particle/{2}", PortFinder.FreeTcpPort(), nodeId, particle.Id)));
            return particle;
        }

        public void Open()
        {
            if(_host.State != CommunicationState.Opened)
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
                return new ParticleState(
                    new double[1],PsoServiceLocator.Instance.GetService<IOptimization<double[]>>().WorstValue(1) );
            }
            try
            {
                var s = _particleClient.GetBestState();
                _particleService.UpdateBestState(s);
            }
            catch
            {
                if (CommunicationBreakdown != null) CommunicationBreakdown();
                Debug.WriteLine("{0} cannot connect to: {1}", Address, RemoteAddress);
            }

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

        public override void UpdateNeighborhood(IParticle[] allParticles)
        {
            Neighborhood = allParticles.Where(p => p.Id != Id).ToArray();
        }

        public override void Init(ParticleState state, double[] velocity, Tuple<double, double>[] bounds = null)
        {

        }

        public override int Id
        {
            get { return _id; }
        }

        public override void UpdateVelocity()
        {
        }

        public override void Transpose(IFitnessFunction<double[], double[]> function)
        {
        }

        public ParticleState PersonalBest
        {
            get { return GetRemoteBest(); }
        }
    }
}