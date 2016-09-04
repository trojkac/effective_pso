using Common;

namespace NetworkManager
{
    public class RemoteCalculationsFinishedHandlerArgs
    {
        public NetworkNodeInfo Source;
        public object Result;

        public RemoteCalculationsFinishedHandlerArgs(NetworkNodeInfo source, object result)
        {
            Source = source;
            Result = result;
        }
    }
}