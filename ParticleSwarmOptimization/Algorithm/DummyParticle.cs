using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Algorithm
{
    class DummyParticle : Particle
    {
        public override int Id
        {
            get { return _id; }
        }

        public override void UpdateVelocity(Common.IState<double[], double[]> globalBest)
        {
            
        }

        public override void UpdateNeighborhood(IParticle[] allParticles)
        {
   
        }

        public override void Transpose(Common.IFitnessFunction<double[], double[]> function)
        {
            var fitness = function.Evaluate(CurrentState.Location);
        }
    }
}
