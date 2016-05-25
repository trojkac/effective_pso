using System;

namespace ManagedGPU
{
    public interface ICudaFitnessFunction
    {
        Func<double[], double> HostFitnessFunction { get; }
        string KernelFile { get; }
    }
}
