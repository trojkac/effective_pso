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
                if (value.FitnessValue < _bestKnownState.FitnessValue)
                {
                    _bestKnownState = value;
                }
            }
        }

        public ParticleService()
        {
            _bestKnownState = ParticleState.WorstState(1);
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
