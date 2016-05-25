using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Algorithm;
using Common;

namespace Controller
{
    public delegate void CalculationCompletedHandler(IState<double[],double[]> result);
    public class PsoController : IPsoController
    {
        private ulong _nodeId;
        public PsoController(ulong nodeId)
        {
            _nodeId = nodeId;
        }

        public event CalculationCompletedHandler CalculationsCompleted;
        public bool CalculationsRunning { get { return RunningAlgorithm != null && !RunningAlgorithm.IsCompleted; } }
        public Task<ParticleState> RunningAlgorithm { get; private set; }
        public PsoSettings RunningSettings { get; private set; }
        private static List<IParticle> CreateParticles(IEnumerable<Tuple<PsoParticleType, int>> particlesParameters, IFitnessFunction<double[],double[]> function, int dimensions, Tuple<double,double>[] bounds)
        {
            var particles = new List<IParticle>();
            foreach (var particleTuple in particlesParameters)
            {
                for (int i = 0; i < particleTuple.Item2; i++)
                {
                    var p = ParticleFactory.Create(particleTuple.Item1, dimensions, 1, function, bounds);
                    particles.Add(p);
                }

            }
            return particles;
        }


        public void Run(PsoSettings psoSettings, PsoService.ProxyParticle[] proxyParticleServices = null)
        {

            var function = AbstractFitnessFunction.GetFitnessFunction(psoSettings.FunctionParameters);
            var particles = CreateParticles(psoSettings.Particles, function, psoSettings.Dimensions, psoSettings.FunctionParameters.SearchSpace);
            if (proxyParticleServices != null)
            {
                particles.AddRange(proxyParticleServices);
            }
            var algorithm = new PsoAlgorithm(psoSettings, function, particles.ToArray());
            RunningAlgorithm = Task<ParticleState>.Factory.StartNew(delegate
            {
                RunningSettings = psoSettings;
                //var r = algorithm.Run(particles,_nodeId.ToString());
                var r = algorithm.Run();
                if (CalculationsCompleted != null) CalculationsCompleted(r);

                return (ParticleState)r;
            });
        }


        public PsoImplementationType[] GetAvailableImplementationTypes()
        {
            throw new NotImplementedException();
        }
    }
}
