using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Common;

namespace Algorithm
{
    public class StandardParticle : Particle
    {
        private const double Phi = 1.4;
        private const double Omega = 0.64;

        public override void Init(ParticleState particleState, double[] velocity)
        {
            CurrentState = particleState;
            Velocity = velocity;
        }

        public override void UpdateVelocity()
        {
            var globalBest = PersonalBest;

            // 1. Find global best
            foreach (var particle in Neighborhood)
            {
                if (particle.PersonalBest.IsBetter(PersonalBest))
                {
                    globalBest = particle.PersonalBest;
                }
            }

            // 2.  get vectors o personal and global best
            var toPersonalBest = CurrentState.VectorTo(PersonalBest);
            var toGlobalBest = CurrentState.VectorTo(globalBest);
			
			var phi1 = RandomGenerator.GetInstance().Random.NextDouble()*Phi;
			var phi2 = RandomGenerator.GetInstance().Random.NextDouble()*Phi;
            
            // 3. multiply them by phi1, phi2 both in [0,Phi]
            toPersonalBest = toPersonalBest.Select(x => x*phi1).ToArray();
            toGlobalBest = toGlobalBest.Select(x => x*phi2).ToArray();

            // 4. multiply velocity by Omega and add toGlobalBest and toPersonalBest
            Velocity = Velocity.Select((v, i) => v*Omega + toGlobalBest[i] + toPersonalBest[i]).ToArray();
        }

        public override void UpdatePersonalBest(IFitnessFunction<double[], double[]> function)
        {
            CurrentState.FitnessValue = function.Evaluate(CurrentState.Location);
            PersonalBest = PersonalBest == null || CurrentState.IsBetter(PersonalBest) ? CurrentState : PersonalBest;
        }

        public override void UpdateNeighborhood(IParticle[] allParticles)
        {
            Neighborhood = allParticles.Where(particle => particle.Id != Id).ToArray();
        }

        public override int Id
        {
            get { return _id; }
        }

        //TODO: IMPORTANT: Translate and UpdateBersonalBest should be private and invoked by one public method
        public override void Translate()
        {
            CurrentState.Location = CurrentState.Location.Select((x, i) => x + Velocity[i]).ToArray();
        }
    }
}
