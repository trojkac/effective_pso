using System.Threading.Tasks;
using Common;
using PsoService;

namespace Controller
{
    public interface IPsoController
    {
        bool CalculationsRunning { get; }
        Task<ParticleState> RunningAlgorithm { get; }
        PsoSettings RunningSettings { get; }
        void Run(PsoSettings psoSettings, ProxyParticleCommunication[] proxyParticleCommunicationServices = null);
        PsoImplementationType[] GetAvailableImplementationTypes();
    }
}