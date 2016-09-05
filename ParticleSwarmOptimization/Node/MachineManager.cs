using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Common;
using Common.Parameters;

namespace Node
{
    public class MachineManager
    {
        private VCpuManager[] _vCpuManagers;
        private GpuManager _gpuManager;



        public MachineManager(string machineIp, int[] ports, int vcpus = -1, bool isGpu = false)
        {
            //_functionParams = functionParams;
            vcpus = vcpus == -1 ? System.Environment.ProcessorCount : vcpus;
            if (ports.Length < vcpus)
            {
                ports = ports.Concat(Enumerable.Range(PortFinder.FreeTcpPort(), vcpus - ports.Length)).ToArray();
            }

            _vCpuManagers = new VCpuManager[vcpus];
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
