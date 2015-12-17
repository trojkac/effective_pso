﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Common;
using ParticleSwarmOptimizationWrapper;
using PsoService;

namespace Controller
{
    class Controller : IPsoController
    {
        private List<Particle> CreateParticles(Tuple<PsoParticleType, int>[] particlesParameters, int dimenstions)
        {
            var particles = new List<Particle>();
            foreach (var particleTuple in particlesParameters)
            {
                switch (particleTuple.Item1)
                {
                    case PsoParticleType.FullyInformed:
                    case PsoParticleType.Standard:
                        particles.AddRange(Enumerable.Repeat(new StandardParticle(dimenstions), particleTuple.Item2));
                        break;
                }
            }
            return particles;
        }
        public ParticleState Run(FitnessFunction fitnessFunction, PsoSettings psoSettings)
        {
            
            var algorithm = PSOAlgorithm.GetAlgorithm(psoSettings.Iterations, fitnessFunction);

            var particles = CreateParticles(psoSettings.Particles, psoSettings.Dimensions);
            return algorithm.Run(particles);
        }

        public ParticleState Run(FitnessFunction fitnessFunction, PsoSettings psoSettings, ProxyParticleService[] proxyParticleServices)
        {
            var algorithm = PSOAlgorithm.GetAlgorithm(psoSettings.Iterations, fitnessFunction);
            var particles = CreateParticles(psoSettings.Particles, psoSettings.Dimensions);
            particles.AddRange(proxyParticleServices.Select(p => new ProxyParticle(psoSettings.Dimensions,p)));
           
            return algorithm.Run(particles);
        }

        public PsoImplementationType[] GetAvailableImplementationTypes()
        {
            throw new NotImplementedException();
        }
    }
}