using System.Collections.Generic;
using Common;

namespace NetworkManager
{
    public interface IReconnaissance
    {
        List<NetworkNodeInfo> KnownNodes { get; } 
    }
}