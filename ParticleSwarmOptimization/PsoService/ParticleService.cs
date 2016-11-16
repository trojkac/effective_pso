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
                if (PsoServiceLocator.Instance.GetService<IOptimization<double[]>>().IsBetter(_bestKnownState.FitnessValue,value.FitnessValue) > 0)
                {
                    _bestKnownState = value;
                }
            }
        }

        public ParticleService()
        {
            _bestKnownState =
                new ParticleState(new double[1], PsoServiceLocator.Instance.GetService<IOptimization<double[]>>().WorstValue(1));
        }

        public ParticleState GetBestState()
        {
            return BestKnownState;
        }

        public void UpdateBestState(ParticleState state)
        {
            BestKnownState = state;
        }

        public void RestartState()
        {
            _bestKnownState =
                new ParticleState(new double[1], PsoServiceLocator.Instance.GetService<IOptimization<double[]>>().WorstValue(1));

        }
    }
}
