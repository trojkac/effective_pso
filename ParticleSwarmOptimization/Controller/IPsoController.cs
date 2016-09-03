using System.Threading.Tasks;
using Common;
using Common.Parameters;
using NetworkManager;
using PsoService;

namespace Controller
{
    public interface IPsoController
    {
        bool CalculationsRunning { get; }
        Task<ParticleState> RunningAlgorithm { get; }
        PsoParameters RunningParameters { get; }
        void Run(PsoParameters psoParameters, ProxyParticle[] proxyParticleServices = null);
        PsoImplementationType[] GetAvailableImplementationTypes();
        event CalculationCompletedHandler CalculationsCompleted;
        void RemoteControllerFinished(RemoteCalculationsFinishedHandlerArgs args);
    }
}