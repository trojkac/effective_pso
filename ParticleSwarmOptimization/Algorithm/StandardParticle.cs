using System;
using System.Linq;
using Common;

namespace Algorithm
{
    public class StandardParticle : Particle
    {
        private const double Phi = 1.4;
        private const double Omega = 0.64;
        private int sinceLastImprovement = 0;
        private const int iterationsToRestart = 10;

        private void Clamp()
        {
            if (Bounds == null || CurrentState == null || CurrentState.Location == null) return;
            CurrentState.Location = CurrentState.Location.Select((x, i) =>  Math.Min(Math.Max(x, Bounds[i].Item1), Bounds[i].Item2)).ToArray();
        }

        public override void Init(ParticleState particleState, double[] velocity, Tuple<double, double>[] bounds = null)
        {
            CurrentState = particleState;
            Velocity = velocity;
            Bounds = bounds;

        }

        public override void UpdateVelocity()
        {
            var globalBest = PersonalBest;

            // 1. Find global best
            foreach (var particle in Neighborhood)
            {
                if (particle.PersonalBest.IsBetter(PersonalBest))
                {
                    globalBest = (ParticleState)particle.PersonalBest.Clone();
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
            if (sinceLastImprovement == iterationsToRestart)
            {
                Init(ParticleStateFactory.Create(
                    CurrentState.Location.Length, CurrentState.FitnessValue.Length),
                    RandomGenerator.GetInstance().RandomVector(CurrentState.Location.Length, -5, 5),
                    Bounds);
                sinceLastImprovement = 0;
            }
            CurrentState.FitnessValue = function.Evaluate(CurrentState.Location);
            // TODO: CurrentState przerobić na typ prosty
            var oldBest = PersonalBest;

            if (PersonalBest == null || CurrentState.IsBetter(PersonalBest))
            {
                PersonalBest = (ParticleState) CurrentState.Clone();
                sinceLastImprovement = 0;
            }
            if (oldBest != null && oldBest.IsCloseToValue(PersonalBest.FitnessValue, 1e-5))
            {
                sinceLastImprovement++;
            }
         
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
            Clamp();
        }
    }
}
