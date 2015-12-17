using System;
using System.Runtime.Serialization;
using System.ServiceModel;

namespace Common
{
    [DataContract]
    public class NetworkNodeInfo : IComparable<NetworkNodeInfo>
    {
        private const int M = 100;
        private static int _lowestAvailableId;
        
        [DataMember]
        public int Id;

        [DataMember]
        public EndpointAddress Address;  //base address in most (all?) cases

        public NetworkNodeInfo(EndpointAddress endpointAddress)
        {
            Address = endpointAddress;
            Id = _lowestAvailableId;
            ++_lowestAvailableId;
        }

        public int Distance(NetworkNodeInfo from)  //d(from.Id, Id)
        {
            return Math.Abs((Id - from.Id)%M);  //modulo?
        }

        public static int Distance(NetworkNodeInfo from, NetworkNodeInfo to)
        {
            return Math.Abs((to.Id - from.Id)%M);
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

        public NetworkNodeInfo(int id, EndpointAddress address)
        {
            if (id < 0 || id >= M)
                throw new ArgumentOutOfRangeException(String.Format("Id should be a value between 0 and {0}",M-1));
            Id = id;
            Address = address;
        }

        public int CompareTo(NetworkNodeInfo obj)
        {
            return Id - obj.Id;
        }
    }
}
