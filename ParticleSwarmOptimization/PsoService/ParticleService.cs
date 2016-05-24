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
                if (PsoServiceLocator.Instance.GetService<IOptimization<double[]>>().IsBetter(_bestKnownState.FitnessValue,value.FitnessValue) > 1)
                {
                    _bestKnownState = value;
                }
            }
        }

        public ParticleService()
        {
            _bestKnownState =
                new ParticleState(PsoServiceLocator.Instance.GetService<IOptimization<double[]>>().WorstValue(1),
                    new double[1]);
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
