using System.Collections.Generic;

namespace Common
{
    public class UserNodeParameters
    {
        public int NrOfVCpu { get; set; }
        public bool IsGpu { get; set; }
        public List<int> Ports { get; set; }
        public List<string> Pipes { get; set; }
        public List<string> PeerAddresses { get; set; }
    }
}
