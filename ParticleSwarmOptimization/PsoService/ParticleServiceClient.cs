using System;
using System.Diagnostics;
using System.ServiceModel;
using System.ServiceModel.Channels;
using Common;

namespace PsoService
{
    public class ParticleServiceClient : ClientBase<IParticleService>, IParticleService
    {
        public ParticleServiceClient(string endpointConfigurationName, string address)
            : base(endpointConfigurationName, address)
        {
        }

        public ParticleServiceClient(Binding binding, EndpointAddress address)
            : base(binding, address)
        {
        }

        public ParticleState GetBestState()
        {
            return base.Channel.GetBestState();            
        }

        public void UpdateBestState(ParticleState state)
        {
            throw new System.NotImplementedException();
        }
        public void RestartState()
        {
            throw new NotImplementedException();
        }

        public static IParticleService CreateClient(string remoteNeighborAddress)
        {
            var binding = new NetTcpBinding(SecurityMode.None);
            var endpoint = new EndpointAddress(remoteNeighborAddress);
            binding.ReceiveTimeout = new TimeSpan(0, 0, 1);
            binding.SendTimeout = new TimeSpan(0, 0, 1);

            return new ParticleServiceClient(binding, endpoint);
        }
    }
}