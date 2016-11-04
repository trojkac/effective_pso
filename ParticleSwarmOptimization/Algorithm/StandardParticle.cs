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

        public StandardParticle(double restartEpsilon, int iterationsToRestart) : base(restartEpsilon,iterationsToRestart)
        {
        }
     


        public override void UpdateVelocity(IState<double[], double[]> globalBest)
        {
            // 1.  get vectors o personal and global best
            var toPersonalBest = Metric.VectorBetween(CurrentState.Location,PersonalBest.Location);
            var toGlobalBest = Metric.VectorBetween(CurrentState.Location, globalBest.Location);
			
			var phi1 = RandomGenerator.GetInstance().RandomVector(CurrentState.Location.Length,0,Phi);
            var phi2 = RandomGenerator.GetInstance().RandomVector(CurrentState.Location.Length, 0, Phi);
            
           
            // 2. multiply velocity by Omega and add toGlobalBest and toPersonalBest
            Velocity = Velocity.Select((v, i) => v*Omega + phi1[i] * toGlobalBest[i] + phi2[i] * toPersonalBest[i]).ToArray();
        }
        public override void UpdateNeighborhood(IParticle[] allParticles)
        {
            Neighborhood = allParticles.Where(particle => particle.Id != Id).ToArray();
            //allParticles.Where(particle => particle.Id == (Id+1)%allParticles.Length || (Id + allParticles.Length - 1)%allParticles.Length == particle.Id).ToArray();
        }
        public override int Id
        {
            get { return _id; }
        }
    }
}