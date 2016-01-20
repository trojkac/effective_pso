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
        private ulong _nodeId;
        public PsoController(ulong nodeId)
        {
            CalculationsRunning = false;
            _nodeId = nodeId;
        }

        public event CalculationCompletedHandler CalculationsCompleted;
        public bool CalculationsRunning { get; private set; }
        public Task<ParticleState> RunningAlgorithm { get; private set; }

        private static List<Particle> CreateParticles(IEnumerable<Tuple<PsoParticleType, int>> particlesParameters, int dimensions)
        {
            var particles = new List<Particle>();
            foreach (var particleTuple in particlesParameters)
            {
                for (int i = 0; i < particleTuple.Item2; i++)
                {
                    Particle p;
                    switch (particleTuple.Item1)
                    {
                        case PsoParticleType.FullyInformed:
                        case PsoParticleType.Standard:
                        default:
                            p = new StandardParticle(dimensions);
                            break;
                    }
                    particles.Add(p);
                }

            }
            return particles;
        }


        public void Run(PsoSettings psoSettings)
        {
            var function = AbstractFitnessFunction.GetFitnessFunction(psoSettings.FunctionParameters);
            var algorithm = PSOAlgorithm.GetAlgorithm(psoSettings.Iterations, function.Calculate);
            var particles = CreateParticles(psoSettings.Particles, psoSettings.Dimensions);
            if (psoSettings.FunctionParameters.SearchSpace != null)
            {
                foreach (var particle in particles)
                {
                    particle.bounds(psoSettings.FunctionParameters.SearchSpace.ToList());
                }   
            }
            RunningAlgorithm = Task<ParticleState>.Factory.StartNew(delegate
            {
                CalculationsRunning = true;
                //var r = algorithm.Run(particles, _nodeId.ToString());
                var r = algorithm.Run(particles);
                if (CalculationsCompleted != null) CalculationsCompleted(r);
                return r;
            });
        }

        public void Run(PsoSettings psoSettings, PsoService.ProxyParticle[] proxyParticleServices)
        {

            var function = AbstractFitnessFunction.GetFitnessFunction(psoSettings.FunctionParameters);
            var algorithm = PSOAlgorithm.GetAlgorithm(psoSettings.Iterations, function.Calculate);
            var particles = CreateParticles(psoSettings.Particles, psoSettings.Dimensions);
            particles.AddRange(proxyParticleServices.Select(p => new ParticleSwarmOptimizationWrapper.ProxyParticle(psoSettings.Dimensions, p)));
            if (psoSettings.FunctionParameters.SearchSpace != null)
            {
                foreach (var particle in particles)
                {
                    particle.bounds(psoSettings.FunctionParameters.SearchSpace.ToList());
                }
            }
            RunningAlgorithm = Task<ParticleState>.Factory.StartNew(delegate
            {
                CalculationsRunning = true;
                //var r = algorithm.Run(particles,_nodeId.ToString());
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
