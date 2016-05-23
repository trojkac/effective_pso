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

        public StandardParticle(IOptimization<double[]> optimization, IMetric<double[]> metric) : base(optimization, metric)
        {
            sinceLastImprovement = 0;
        }

        private double[] GetClampedLocation(double[] vector)
        {
            if (Bounds == null || vector == null) return vector;
            return vector.Select((x, i) =>  Math.Min(Math.Max(x, Bounds[i].Item1), Bounds[i].Item2)).ToArray();
        }

        public override void Init(ParticleState particleState, double[] velocity, Tuple<double, double>[] bounds = null)
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
                if (_optimization.IsBetter(particle.PersonalBest.FitnessValue,PersonalBest.FitnessValue)<0)
                {
                    globalBest = particle.PersonalBest;
                }
            }

            // 2.  get vectors o personal and global best
            var toPersonalBest = _metric.VectorBetween(CurrentState.Location,PersonalBest.Location);
            var toGlobalBest = _metric.VectorBetween(CurrentState.Location, globalBest.Location);
			
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

            if (_optimization.IsBetter(newVal,PersonalBest.FitnessValue) < 0)
            {
                PersonalBest = CurrentState;
                sinceLastImprovement = 0;
            }
            if (_optimization.AreClose(oldBest.FitnessValue,PersonalBest.FitnessValue, 1e-5))
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
        }
    }
}
