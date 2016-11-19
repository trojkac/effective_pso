using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Algorithm;
using Common;
using Common.Parameters;
using NetworkManager;
using ManagedGPU;

namespace Controller
{
    public delegate void CalculationCompletedHandler(ParticleState result);
    public class PsoController : IPsoController
    {
        private ulong _nodeId;
        IFitnessFunction<double[], double[]> _function;
        private CancellationTokenSource _tokenSource;

        public PsoController(ulong nodeId)
        {
            _nodeId = nodeId;
            _tokenSource = new CancellationTokenSource();
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

        public ParticleState Stop()
        {
            stopCalculationsAndPrepareToken();
            return (ParticleState) _function.BestEvaluation;
        }

        private void stopCalculationsAndPrepareToken()
        {
            _tokenSource.Cancel();
            RunningAlgorithm.Wait();
            _tokenSource.Dispose();
            _tokenSource = new CancellationTokenSource();
        }

        public bool CalculationsRunning { get { return RunningAlgorithm != null && !RunningAlgorithm.IsCompleted; } }
        public Task<ParticleState> RunningAlgorithm { get; private set; }
        public PsoParameters RunningParameters { get; private set; }
        private static List<IParticle> CreateParticles(PsoParameters parameters, IFitnessFunction<double[],double[]> function)
        {
            var particles = new List<IParticle>();
            foreach (var particle in parameters.Particles)
            {
                for (int i = 0; i < particle.Count; i++)
                {
                    var p = ParticleFactory.Create(particle.ParticleType, parameters.FunctionParameters.Dimension, 1, function, parameters.Epsilon, parameters.ParticleIterationsToRestart, parameters.FunctionParameters.SearchSpace);
                    particles.Add(p);
                }

            }
            return particles;
        }


        public void Run(PsoParameters psoParameters, IParticle[] proxyParticleServices = null)
        {
            _function = FunctionFactory.GetFitnessFunction(psoParameters.FunctionParameters);

            var parts = psoParameters.FunctionParameters.FitnessFunctionType.Split('_');
            var functionNr = Int32.Parse(parts[1].Substring(1));
            var instanceStr = parts[2];
            var instanceNr = Int32.Parse(instanceStr.Substring(1));

            var gpu = GpuController.Setup(
                new CudaParams
                {
                    FitnessFunction = _function,
                    FitnessDimensions = 1,
                    LocationDimensions = psoParameters.FunctionParameters.Dimension,
                    FunctionNumber = functionNr,
                    InstanceNumber = instanceNr,
                    Iterations = 5000,
                    ParticlesCount = 640,
                    SyncWithCpu = true
                });

            var cudaAlgorithm = gpu.Item2;
            var cudaParticle = gpu.Item1;

            _function = FunctionFactory.GetFitnessFunction(psoParameters.FunctionParameters);
            var particles = CreateParticles(psoParameters, _function);
            if (proxyParticleServices != null)
            {
                particles.AddRange(proxyParticleServices);
            }

            if (functionNr != 21 && functionNr != 22)
                particles.Add(cudaParticle);

            var token = _tokenSource.Token;
            var algorithm = new PsoAlgorithm(psoParameters, _function, particles.ToArray());
            RunningAlgorithm = Task<ParticleState>.Factory.StartNew(delegate
            {
                RunningParameters = psoParameters;
                //var r = algorithm.Run(particles,_nodeId.ToString());

                if (functionNr != 21 && functionNr != 22)
                    cudaAlgorithm.RunAsync();
                var r = algorithm.Run();
                if (CalculationsCompleted != null) CalculationsCompleted((ParticleState)r);

                if (functionNr != 21 && functionNr != 22)
                    cudaAlgorithm.Wait();
                return (ParticleState)r;
            }, token);
        }


        public PsoImplementationType[] GetAvailableImplementationTypes()
        {
            throw new NotImplementedException();
        }

        public void UpdateResultWithOtherNodes(ParticleState[] bestFrmOtherNodes)
        {

            foreach (var particleState in bestFrmOtherNodes)
            {
                if (particleState.Location == null) continue;
                _function.Evaluate(particleState.Location);
            }
        }
    }
}
