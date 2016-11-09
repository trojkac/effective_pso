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
            if(Neighborhood == null)
            {
                Neighborhood = allParticles.Where(particle => particle.Id != Id).ToArray();
            }
        }

        public override void Transpose(Common.IFitnessFunction<double[], double[]> function)
        {
            var best = CurrentState;
            if(Neighborhood != null)
            {
                foreach (var particle in Neighborhood)
                {
                    if (Optimization.IsBetter(best.FitnessValue, particle.PersonalBest.FitnessValue) > 0)
                        best = particle.PersonalBest;
                }
            }
            


            var fitness = function.Evaluate(best.Location);
            CurrentState = new Common.ParticleState(best.Location, best.FitnessValue);
            PersonalBest = new Common.ParticleState(best.Location, best.FitnessValue);

        }
    }
}
