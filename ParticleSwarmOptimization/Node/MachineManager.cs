using System.Collections.Generic;

namespace Node
{
    public class MachineManager
    {
        private List<VCpuManager> vCpuManagers;
        private GpuManager gpuManager;

        public MachineManager()
        {
            vCpuManagers = new List<VCpuManager>();
        }
    }
}
