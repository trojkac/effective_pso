using Algorithm;
using Common;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.ServiceModel;
using System.Text;
using System.Threading.Tasks;

namespace PsoService
{
    public class ProxyManager
    {
        public event ParticleCommunicationBreakdown CommunicationBreakdown;


        private ServiceHost _host;
        private int _communicationErrorCount;
        private int _communicationErrorLimit = 10;
        private IParticleService _particleClient;
        private IParticleService _particleService;


        public Uri RemoteAddress { get; private set; }

        public Uri Address
        {
            get
            {
                return _host.BaseAddresses.First();
            }
        }
        public NetworkNodeInfo RemoteNode
        {
            get
            {
                return new NetworkNodeInfo(RemoteAddress.Authority, "");
            }
        }

        public ProxyManager(ulong nodeId, int particleId)
        {
            _particleService = new ParticleService();
            _host = new ServiceHost(_particleService,
                new Uri(string.Format("net.tcp://0.0.0.0:{0}/{1}/particle/{2}", PortFinder.FreeTcpPort(), nodeId, particleId))
                );
            _host.AddServiceEndpoint(typeof(IParticleService), new NetTcpBinding(SecurityMode.None), "");

        }


        public void Open()
        {
            if (_host.State != CommunicationState.Opened)
                _host.Open();
        }

        public void Close()
        {
            _host.Close();
        }
        public void UpdateRemoteAddress(Uri address)
        {
            RemoteAddress = address;
            _particleClient = ParticleServiceClient.CreateClient(address.ToString());
        }
        public void RestartState()
        {
            _particleService.RestartState();
        }
        internal ParticleState GetRemoteBestState()
        {
            if (_particleClient == null)
            {
                return new ParticleState();
            }
            try
            {
                var s = _particleClient.GetBestState();
                _particleService.UpdateBestState(s);
                _communicationErrorCount = 0;
                return s;
            }
            catch
            {
                _communicationErrorCount++;
                if (CommunicationBreakdown != null && _communicationErrorCount >= _communicationErrorLimit)
                {
                    CommunicationBreakdown();
                    Debug.WriteLine("{0} cannot connect to: {1}", Address, RemoteAddress);
                }
                return new ParticleState();
            }
        }
        public ParticleState GetBestState()
        {
            return _particleService.GetBestState();
        }
        public void UpdateBestState(ParticleState state)
        {
            _particleService.UpdateBestState(state);
        }
    }
}
