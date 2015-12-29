using System;
using System.Runtime.Serialization;
using System.ServiceModel;

namespace NetworkManager
{
    [DataContract]
    public class NetworkNodeInfo : IComparable<NetworkNodeInfo>
    {
        private const int M = 100;
        private static int _lowestAvailableId;

        [DataMember]
        public int Id;

        [DataMember]
        public EndpointAddress TcpAddress;

        [DataMember]
        public EndpointAddress PipeAddress;

        public NetworkNodeInfo()
        {
            Id = _lowestAvailableId++;
        }

        public NetworkNodeInfo(string tcpAddress, string pipeAddress)
            : this()
        {
            TcpAddress = new EndpointAddress(tcpAddress);
            PipeAddress = new EndpointAddress(pipeAddress);
        }

        public NetworkNodeInfo(int id, string tcpAddress, string pipeAddress)
        {
            if (id < 0 || id >= M)
                throw new ArgumentOutOfRangeException(String.Format("Id should be a value between 0 and {0}", M - 1));

            Id = id;
            TcpAddress = new EndpointAddress(tcpAddress);
            PipeAddress = new EndpointAddress(pipeAddress);
        }

        public int Distance(NetworkNodeInfo from)  //d(from.Id, Id)
        {
            return Math.Abs((Id - from.Id) % M);  //modulo?
        }

        public static int Distance(NetworkNodeInfo from, NetworkNodeInfo to)
        {
            return Math.Abs((to.Id - from.Id) % M);
        }

        public static bool operator <(NetworkNodeInfo x, NetworkNodeInfo y)
        {
            return x.Id < y.Id;  //modulo?
        }

        public static bool operator >(NetworkNodeInfo x, NetworkNodeInfo y)
        {
            return x.Id > y.Id;  //modulo?
        }

        public static int operator -(NetworkNodeInfo x, NetworkNodeInfo y)
        {
            return x.Id - y.Id;  //modulo?
        }

        public int CompareTo(NetworkNodeInfo obj)
        {
            return Id - obj.Id;
        }
    }
}