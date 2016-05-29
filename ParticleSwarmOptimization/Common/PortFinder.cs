using System.Net;
using System.Net.Sockets;

namespace Common
{
    public static class PortFinder
    {
        private static int _basePort = 3000;
        private static int _cnt = 0;
        public static int FreeTcpPort()
        {
            return _basePort + _cnt++;
        }
    }
}