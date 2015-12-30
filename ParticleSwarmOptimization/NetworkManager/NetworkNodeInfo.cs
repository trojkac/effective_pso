using System;
using System.Runtime.Serialization;

namespace NetworkManager
{
    [DataContract]
    public class NetworkNodeInfo : IComparable<NetworkNodeInfo>, IEquatable<NetworkNodeInfo>
    {
        [DataMember]
        private const int M = 100;
        private static int _lowestAvailableId;

        [DataMember]
        public int Id;

        [DataMember]
        public string TcpAddress;

        [DataMember]
        public string PipeAddress;

        public NetworkNodeInfo()
        {
            Id = _lowestAvailableId++;
        }

        public NetworkNodeInfo(string tcpAddress, string pipeAddress)
            : this()
        {
            TcpAddress = tcpAddress;
            PipeAddress = pipeAddress;
        }

        public NetworkNodeInfo(int id, string tcpAddress, string pipeAddress)
        {
            if (id < 0 || id >= M)
                throw new ArgumentOutOfRangeException(String.Format("Id should be a value between 0 and {0}", M - 1));

            Id = id;
            TcpAddress = tcpAddress;
            PipeAddress = pipeAddress;
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

        public bool Equals(NetworkNodeInfo other)
        {
            return TcpAddress.Equals(other.TcpAddress) && PipeAddress.Equals(other.PipeAddress);
        }

        public override int GetHashCode()
        {
            return (TcpAddress + PipeAddress).GetHashCode();
        }
    }
}