using Common;
using ParticleSwarmOptimizationWrapper;

namespace Controller
{
    public interface IPsoController
    {
        double Run(FitnessFunction fitnessFunction, PsoSettings psoSettings);
        PsoImplementationType[] GetAvailableImplementationTypes();
    }
}