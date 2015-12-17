using Common;
using PsoService;

namespace Controller
{
    public interface IPsoController
    {
        ParticleState  Run(FitnessFunction fitnessFunction, PsoSettings psoSettings);
        ParticleState Run(FitnessFunction fitnessFunction, PsoSettings psoSettings, ProxyParticleService[] proxyParticleServices);
        PsoImplementationType[] GetAvailableImplementationTypes();
    }
}