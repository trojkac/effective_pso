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
        private CancellationTokenSource _cudaTokenSource;

        public PsoController(ulong nodeId)
        {
            _nodeId = nodeId;
            _tokenSource = new CancellationTokenSource();
            _cudaTokenSource = new CancellationTokenSource();
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
        private Task<double> RunningCudaAlgorithm { get; set; }
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

        PsoAlgorithm _algorithm;
        GenericCudaAlgorithm _cudaAlgorithm;

        public void Run(PsoParameters psoParameters, IParticle[] proxyParticleServices = null)
        {
            _function = FunctionFactory.GetFitnessFunction(psoParameters.FunctionParameters);
            var cudaParticle = PrepareCudaAlgorithm(psoParameters);            
            var particles = PrepareParticles(psoParameters,proxyParticleServices,cudaParticle);
            RunningParameters = psoParameters;
            _algorithm = new PsoAlgorithm(psoParameters, _function, particles.ToArray());


            RunningCudaAlgorithm = Task<double>.Factory.StartNew(() =>
            {
                var result = _cudaAlgorithm.Run(_cudaTokenSource.Token);
                _cudaAlgorithm.Dispose();
                return result;

            }, _cudaTokenSource.Token);
            RunningAlgorithm = Task<ParticleState>.Factory.StartNew(delegate
            {
                return StartAlgorithm(_tokenSource.Token);
            }, _tokenSource.Token);
        }
        private CudaParticle PrepareCudaAlgorithm(PsoParameters psoParameters)
        {
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

            _cudaAlgorithm = gpu.Item2;

            return gpu.Item1;
        }

        private List<IParticle> PrepareParticles(PsoParameters psoParameters, IParticle[] proxyParticleServices, CudaParticle cudaParticle)
        {
            var particles = CreateParticles(psoParameters,_function);
            if (proxyParticleServices != null)
            {
                particles.AddRange(proxyParticleServices);
            }
            if(cudaParticle != null)
            {
                particles.Add(cudaParticle);
            }
            return particles;
        }

        private ParticleState StartAlgorithm(CancellationToken token)
        {
            
            var r = _algorithm.Run(token);
            StopCudaCalculations();
            if (CalculationsCompleted != null) CalculationsCompleted((ParticleState)r);
            return (ParticleState)r;
        }

        private void StopCudaCalculations()
        {

            _cudaTokenSource.Cancel();
            RunningCudaAlgorithm.Wait();
            _cudaTokenSource.Dispose();
            _cudaTokenSource = new CancellationTokenSource();
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
