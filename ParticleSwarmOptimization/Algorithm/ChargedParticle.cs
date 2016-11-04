using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Common;

namespace Algorithm
{
    class ChargedParticle : Particle
    {
        
        private readonly double _charge;
        private double _rcore;
        private double _rlimit;
        public double Charge { get { return _charge;} }
        public ChargedParticle(double charge = -1, double rcore = -1, double rlimit = -1)
        {
            _charge = charge > 0 ? charge : Constants.CHARGE;
            _rcore = rcore > 0 ? rcore : Constants.REPULSION_CORE;
            _rlimit = rlimit > 0 ? rlimit : Constants.REPULSION_LIMIT;
        }

        public override int Id
        {
            get { return _id; }
        }

        public override void UpdateVelocity(IState<double[], double[]> globalBest)
        {
            // 1.  get vectors o personal and global best
            var toPersonalBest = Metric.VectorBetween(CurrentState.Location, PersonalBest.Location);
            var toGlobalBest = Metric.VectorBetween(CurrentState.Location, globalBest.Location);

            var phi1 = RandomGenerator.GetInstance().RandomVector(CurrentState.Location.Length, 0, Constants.PHI);
            var phi2 = RandomGenerator.GetInstance().RandomVector(CurrentState.Location.Length, 0, Constants.PHI);

            // 2. acceleration
            var  acceleration = Enumerable.Repeat(0.0, CurrentState.Location.Length).ToArray();
            foreach (var particle in Neighborhood)
            {
                if (!(particle is ChargedParticle)) continue;
                var vector = Metric.VectorBetween(CurrentState.Location, particle.CurrentState.Location);
                var dist = Metric.Norm(vector);
                if (!(dist >= _rcore) || !(dist <= _rlimit)) continue;
                var dist3 = Math.Pow(dist, 3);
                var accel = _charge*((ChargedParticle) particle).Charge/dist3;
                acceleration = acceleration.Select((a, i) => accel * vector[i] + a).ToArray();
            }
            // 2. multiply velocity by Omega and add toGlobalBest and toPersonalBest
            Velocity = Velocity.Select((v, i) => v * Constants.OMEGA + phi1[i] * toGlobalBest[i] + phi2[i] * toPersonalBest[i] + acceleration[i]).ToArray();

            
        }



        public override void UpdateNeighborhood(IParticle[] allParticles)
        {
            Neighborhood = allParticles.Where(particle => particle.Id != Id).ToArray();
            //allParticles.Where(particle => particle.Id == (Id+1)%allParticles.Length || (Id + allParticles.Length - 1)%allParticles.Length == particle.Id).ToArray();
        }

    }
}
