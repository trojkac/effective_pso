using System;
using System.Dynamic;
using System.Linq;
using System.Runtime.InteropServices;
using Common;

namespace Algorithm
{
    public class StandardParticle : Particle
    {
        private const double Phi = 1.4;
        private const double Omega = 0.64;
        private int sinceLastImprovement;
        private const int iterationsToRestart = 10;

        public StandardParticle()
        {
            sinceLastImprovement = 0;
        }

        private double[] GetClampedLocation(double[] vector)
        {
            if (Bounds == null || vector == null) return vector;
            return vector.Select((x, i) =>  Math.Min(Math.Max(x, Bounds[i].Min), Bounds[i].Max)).ToArray();
        }

        public override void Init(ParticleState particleState, double[] velocity, DimensionBound[] bounds = null)
        {
            CurrentState = particleState;
            PersonalBest = particleState;
            Velocity = velocity;
            Bounds = bounds;

        }
      
        public override void UpdateVelocity()
        {
            var globalBest = PersonalBest;

            // 1. Find global best
            foreach (var particle in Neighborhood)
            {
                if (PsoServiceLocator.Instance.GetService<IOptimization<double[]>>().IsBetter(particle.PersonalBest.FitnessValue,PersonalBest.FitnessValue)<0)
                {
                    globalBest = particle.PersonalBest;
                }
            }

            // 2.  get vectors o personal and global best
            var toPersonalBest = PsoServiceLocator.Instance.GetService<IMetric<double[]>>().VectorBetween(CurrentState.Location,PersonalBest.Location);
            var toGlobalBest = PsoServiceLocator.Instance.GetService<IMetric<double[]>>().VectorBetween(CurrentState.Location, globalBest.Location);
			
			var phi1 = RandomGenerator.GetInstance().RandomVector(CurrentState.Location.Length,0,Phi);
            var phi2 = RandomGenerator.GetInstance().RandomVector(CurrentState.Location.Length, 0, Phi);
            
           
            // 4. multiply velocity by Omega and add toGlobalBest and toPersonalBest
            Velocity = Velocity.Select((v, i) => v*Omega + phi1[i] * toGlobalBest[i] + phi2[i] * toPersonalBest[i]).ToArray();
        }

        public override void Transpose(IFitnessFunction<double[], double[]> function)
        {
            double[] newLocation; 
            if (sinceLastImprovement == iterationsToRestart)
            {
                newLocation = RandomGenerator.GetInstance().RandomVector(CurrentState.Location.Length, -5, 5);
                sinceLastImprovement = 0;
            }
            else
            {
                newLocation = GetClampedLocation(CurrentState.Location.Select((x, i) => x + Velocity[i]).ToArray());
            }
            var newVal = function.Evaluate(newLocation);
            var oldBest = PersonalBest;
            CurrentState = new ParticleState(newLocation, newVal);

            if (PsoServiceLocator.Instance.GetService<IOptimization<double[]>>().IsBetter(newVal,PersonalBest.FitnessValue) < 0)
            {
                PersonalBest = CurrentState;
                sinceLastImprovement = 0;
            }
            if (PsoServiceLocator.Instance.GetService<IOptimization<double[]>>().AreClose(oldBest.FitnessValue,PersonalBest.FitnessValue, 1e-5))
            {
                sinceLastImprovement++;
            }
         
        }

        public override void UpdateNeighborhood(IParticle[] allParticles)
        {
            Neighborhood = allParticles.Where(particle => particle.Id == (Id+1)%allParticles.Length || (Id + allParticles.Length - 1)%allParticles.Length == particle.Id).ToArray();
        }

        public override int Id
        {
            get { return _id; }
        }
    }
}
