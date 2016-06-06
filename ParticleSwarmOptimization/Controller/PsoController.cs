using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Algorithm;
using Common;
using NetworkManager;

namespace Controller
{
    public delegate void CalculationCompletedHandler(IState<double[],double[]> result);
    public class PsoController : IPsoController
    {
        private ulong _nodeId;
        IFitnessFunction<double[], double[]> _function;
        public PsoController(ulong nodeId)
        {
            _nodeId = nodeId;
        }

        public event CalculationCompletedHandler CalculationsCompleted;
        public void RemoteControllerFinished(RemoteCalculationsFinishedHandlerArgs args)
        {
            if (CalculationsRunning)
            {
                // evaluating function for best state of remote node - in case it's the best evaluation.
                _function.Evaluate(((ParticleState)args.Result).Location);
            }
        }

        public bool CalculationsRunning { get { return RunningAlgorithm != null && !RunningAlgorithm.IsCompleted; } }
        public Task<ParticleState> RunningAlgorithm { get; private set; }
        public PsoParameters RunningParameters { get; private set; }
        private static List<IParticle> CreateParticles(ParticlesCount[] particlesParameters, IFitnessFunction<double[],double[]> function, int dimensions, Tuple<double,double>[] bounds)
        {
            var particles = new List<IParticle>();
            foreach (var particle in particlesParameters)
            {
                for (int i = 0; i < particle.Count; i++)
                {
                    var p = ParticleFactory.Create(particle.ParticleType, dimensions, 1, function, bounds);
                    particles.Add(p);
                }

            }
            return particles;
        }


        public void Run(PsoParameters psoParameters, PsoService.ProxyParticle[] proxyParticleServices = null)
        {

            _function = FunctionFactory.GetFitnessFunction(psoParameters.FunctionParameters);
            var particles = CreateParticles(psoParameters.Particles, _function, psoParameters.FunctionParameters.Dimension, psoParameters.FunctionParameters.SearchSpace);
            if (proxyParticleServices != null)
            {
                particles.AddRange(proxyParticleServices);
            }
            var algorithm = new PsoAlgorithm(psoParameters, _function, particles.ToArray());
            RunningAlgorithm = Task<ParticleState>.Factory.StartNew(delegate
            {
                RunningParameters = psoParameters;
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
