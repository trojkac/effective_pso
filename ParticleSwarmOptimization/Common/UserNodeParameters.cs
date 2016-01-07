using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common
{
    public class UserNodeParameters
    {
        public int NrOfVCpu { get; set; }
        public bool IsGpu { get; set; }
        public List<string> PeerAddresses { get; set; }
    }
}
