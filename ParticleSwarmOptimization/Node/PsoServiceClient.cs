using System.ServiceModel;
using Common;

namespace Node
{
    public class PsoServiceClient : ClientBase<IPsoService>, IPsoService
    {
        public PsoServiceClient(string endpointConfigurationName, string address)
            : base(endpointConfigurationName, address)
        {
        }

        public PsoServiceClient(EndpointAddress endpoint)
            : base()
        {
            
        }

        public ParticleState GetBestState(int nodeId)
        {
            return base.Channel.GetBestState(nodeId);
        }
    }
}