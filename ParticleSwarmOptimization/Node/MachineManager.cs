using System.Collections.Generic;
using System.Threading.Tasks;
using Common;

namespace Node
{
    public class MachineManager
    {
        private VCpuManager[] _vCpuManagers;
        private GpuManager _gpuManager;



        public MachineManager(string machineIp, int vcpus, int[] ports, bool isGpu = false)
        {
            //_functionParams = functionParams;
            if (ports.Length != vcpus) {
                throw new System.ArgumentException("ports.Length has to be equal to vcpus");
            }

            _vCpuManagers = new VCpuManager[vcpus] ;
            for (int i = 0; i < vcpus; i++)
            {
                _vCpuManagers[i] = new VCpuManager(machineIp, ports[i], i.ToString());
                _vCpuManagers[i].StartTcpNodeService();
                if (i > 0)
                {
                    _vCpuManagers[i].NetworkNodeManager.Register(_vCpuManagers[i - 1].GetMyNetworkNodeInfo());
                }
            }
            if (isGpu)
            {
                _gpuManager = new GpuManager();
            }

          
        }
        public void Register(string remoteAddress)
        {
            foreach (var vcpu in _vCpuManagers)
            {
                vcpu.NetworkNodeManager.Register(new NetworkNodeInfo(remoteAddress, "asd"));
            }
        }
        public void StartPsoAlgorithm(PsoParameters parameters)
        {
            _vCpuManagers[0].StartCalculations(parameters);
        }

        public ParticleState GetResult()
        {
             _vCpuManagers[0].PsoController.RunningAlgorithm.Wait();
             return _vCpuManagers[0].PsoController.RunningAlgorithm.Result;
        }
    }
}
