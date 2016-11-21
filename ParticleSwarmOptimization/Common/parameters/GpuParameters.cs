using System.Runtime.Serialization;
using Common.Parameters;

namespace Common.parameters
{
    public class GpuParameters : IParameters
    {
        [DataMember] public bool UseGpu;

        [DataMember] public int ParticlesCount;

        [DataMember] public int Iterations;
    }
}
