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
            : base(binding,address)
        {
        }
        public ParticleState GetBestState()
        {
            return base.Channel.GetBestState();
        }
    }
}