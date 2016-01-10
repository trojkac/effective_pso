using System.Collections.Generic;

namespace Common
{
    public class UserNodeParameters
    {
        public int NrOfVCpu { get; set; }
        public bool IsGpu { get; set; }
        public List<string> PeerAddresses { get; set; }
    }
}
