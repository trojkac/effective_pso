using System.ServiceModel;
using System.ServiceModel.Channels;
using Common;

namespace Node
{
    public class PsoServiceClient : ClientBase<IPsoService>, IPsoService
    {
        public PsoServiceClient(string endpointConfigurationName, string address)
            : base(endpointConfigurationName, address)
        {
        }
        public PsoServiceClient(Binding binding, EndpointAddress address)
            : base(binding,address)
        {
        }
        public ParticleState GetBestState()
        {
            return base.Channel.GetBestState();
        }
    }
}