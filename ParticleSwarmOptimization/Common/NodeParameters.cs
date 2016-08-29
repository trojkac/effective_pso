using System.Collections.Generic;

namespace Common
{
    public class NodeParameters
    {
        public int NrOfVCpu { get; set; }
        public bool IsGpu { get; set; }
        public string Ip { get; set; }
        public List<int> Ports { get; set; }
        public string PeerAddress { get; set; }
    }
}
