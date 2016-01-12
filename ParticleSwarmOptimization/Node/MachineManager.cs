using System.Collections.Generic;
using System.Threading.Tasks;
using Common;

namespace Node
{
    public class MachineManager
    {
        private List<VCpuManager> _vCpuManagers;
        private GpuManager _gpuManager;

        private UserNodeParameters _nodeParams;
        private UserFunctionParameters _functionParams;
        private UserPsoParameters _psoParams;

        public MachineManager(UserNodeParameters nodeParams, UserFunctionParameters functionParams, UserPsoParameters psoParams)
        {
            _nodeParams = nodeParams;
            _functionParams = functionParams;
            _psoParams = psoParams;

            _vCpuManagers = new List<VCpuManager>();
            for (int i = 0; i < nodeParams.NrOfVCpu; i++)
            {
                _vCpuManagers.Add(new VCpuManager(nodeParams.Ports[i], nodeParams.Pipes[i]));
            }

            if (nodeParams.IsGpu)
            {
                _gpuManager = new GpuManager();
            }
        }

        public void StartPsoAlgorithm()
        {
            PsoSettings psoSettings = new PsoSettings(_psoParams, _functionParams);
            Parallel.ForEach(_vCpuManagers, (cpuMgr) =>
            {
                cpuMgr.Run( psoSettings);
            });
        }
    }
}
