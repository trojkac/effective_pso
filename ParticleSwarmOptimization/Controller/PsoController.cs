using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Common;
using ParticleSwarmOptimizationWrapper;

namespace Controller
{
    public delegate void CalculationCompletedHandler(ParticleState result);
    public class PsoController : IPsoController
    {
        public PsoController()
        {
            CalculationsRunning = false;
        }

        public event CalculationCompletedHandler CalculationsCompleted;
        public bool CalculationsRunning { get; private set; }
        public Task<ParticleState> RunningAlgorithm { get; private set; }

        private static List<Particle> CreateParticles(IEnumerable<Tuple<PsoParticleType, int>> particlesParameters, int dimensions)
        {
            var particles = new List<Particle>();
            foreach (var particleTuple in particlesParameters)
            {
                switch (particleTuple.Item1)
                {
                    case PsoParticleType.FullyInformed:
                    case PsoParticleType.Standard:
                        particles.AddRange(Enumerable.Repeat(new StandardParticle(dimensions), particleTuple.Item2));
                        break;
                }
            }
            return particles;
        }


        public void Run( PsoSettings psoSettings)
        {
            var function = AbstractFitnessFunction.GetFitnessFunction(psoSettings.FunctionParameters);
            var algorithm = PSOAlgorithm.GetAlgorithm(psoSettings.Iterations, function.Calculate);
            var particles = CreateParticles(psoSettings.Particles, psoSettings.Dimensions);
            RunningAlgorithm = Task<ParticleState>.Factory.StartNew(delegate
            {
                CalculationsRunning = true;
                var r = algorithm.Run(particles);
                if (CalculationsCompleted != null) CalculationsCompleted(r);
                return r;
            });
        }

        public void Run( PsoSettings psoSettings, PsoService.ProxyParticle[] proxyParticleServices)
        {
           
            var function = AbstractFitnessFunction.GetFitnessFunction(psoSettings.FunctionParameters);
            var algorithm = PSOAlgorithm.GetAlgorithm(psoSettings.Iterations, function.Calculate);
            var particles = CreateParticles(psoSettings.Particles, psoSettings.Dimensions);
            particles.AddRange(proxyParticleServices.Select(p => new ParticleSwarmOptimizationWrapper.ProxyParticle(psoSettings.Dimensions, p)));
            RunningAlgorithm = Task<ParticleState>.Factory.StartNew(delegate
            {
                CalculationsRunning = true;
                var r = algorithm.Run(particles);
                if (CalculationsCompleted != null) CalculationsCompleted(r);
                return r;
            });
        }


        public PsoImplementationType[] GetAvailableImplementationTypes()
        {
            throw new NotImplementedException();
        }
    }
}
