using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
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
            fibonacci(12);
            CurrentState = new Common.ParticleState(globalBest.Location, globalBest.FitnessValue);
        }

        private int fibonacci(int n)
        {
            if(n == 0 || n == 1)
            {
                return n;
            }
            return fibonacci(n-1)+fibonacci(n-2);
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
