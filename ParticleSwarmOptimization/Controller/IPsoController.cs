using System.Threading.Tasks;
using Common;
using PsoService;

namespace Controller
{
    public interface IPsoController
    {
        bool CalculationsRunning { get; }
        Task<ParticleState> RunningAlgorithm { get; }
        void Run(PsoSettings psoSettings);
        void Run(PsoSettings psoSettings, ProxyParticle[] proxyParticleServices);
        PsoImplementationType[] GetAvailableImplementationTypes();
    }
}