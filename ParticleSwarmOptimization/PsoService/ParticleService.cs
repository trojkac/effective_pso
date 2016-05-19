using System.Diagnostics;
using System.ServiceModel;
using Common;

namespace PsoService
{
    [ServiceBehavior(InstanceContextMode = InstanceContextMode.Single)]
    internal class ParticleService : IParticleService
    {
        private ParticleState _bestKnownState;
        public ParticleState BestKnownState
        {
            get
            {
                return _bestKnownState;
            }
            set
            {
                if (value.IsBetter(_bestKnownState))
                {
                    _bestKnownState = value;
                }
            }
        }

        public ParticleService()
        {

        }

        public ParticleState GetBestState()
        {
            return BestKnownState;
        }

        public void UpdateBestState(ParticleState state)
        {
            BestKnownState = state;
        }
    }
}
