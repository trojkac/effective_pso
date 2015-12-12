using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.ServiceModel;
using System.Text;
using Common;

namespace Service
{
    // NOTE: You can use the "Rename" command on the "Refactor" menu to change the class name "Service" in both code and config file together.
    public class Service : IService
    {
        public string GetKnownBest(int value)
        {
            return string.Format("You entered: {0}", value);
        }


        public ParticleState GetKnownBest()
        {
            return new ParticleState(new[]{0.0,0.0},5.0);
        }

        public ParticleState Run(PsoSettings settings)
        {
            throw new NotImplementedException();
        }
    }
}
