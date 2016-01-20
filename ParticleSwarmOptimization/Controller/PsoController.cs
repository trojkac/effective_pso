using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Common;
using ParticleSwarmOptimizationWrapper;
using CudaPsoWrapper;

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

        private static List<ParticleWrapper> CreateParticles(IEnumerable<Tuple<PsoParticleType, int>> particlesParameters, int dimensions)
        {
            var particles = new List<ParticleWrapper>();
            foreach (var particleTuple in particlesParameters)
            {
                for (int i = 0; i < particleTuple.Item2; i++)
                {
                    ParticleWrapper p;
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
            
            var cudaAlgorithm = CudaPSOAlgorithm.createAlgorithm(psoSettings.Iterations);

            unsafe
            {
                var cudaParticle =
                    CudaPraticleWrapperFactory.Create(cudaAlgorithm.getLocalEndpoint(), cudaAlgorithm.getRemoteEndpoint());
                particles.Add(cudaParticle);
            }

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
                new Task(cudaAlgorithm.run).Start();
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

            var cudaAlgorithm = CudaPSOAlgorithm.createAlgorithm(psoSettings.Iterations);

            unsafe
            {
                var cudaParticle =
                    CudaPraticleWrapperFactory.Create(cudaAlgorithm.getLocalEndpoint(), cudaAlgorithm.getRemoteEndpoint());
                particles.Add(cudaParticle);
            }

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
                new Task(cudaAlgorithm.run).Start();
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
