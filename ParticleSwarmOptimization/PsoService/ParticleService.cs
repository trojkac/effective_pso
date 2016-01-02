﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.ServiceModel;
using System.Text;
using System.Threading.Tasks;
using Common;

namespace PsoService
{
    [ServiceBehavior(InstanceContextMode = InstanceContextMode.Single)]
    class ParticleService : IParticleService
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
                if (value.FitnessValue > _bestKnownState.FitnessValue)
                {
                    _bestKnownState = value;
                }
            }
        }

        public ParticleService()
        {
            _bestKnownState = ParticleState.WorstState;
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
