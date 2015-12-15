using System;
using System.Runtime.Serialization;
using System.ServiceModel;

namespace Node
{
    [DataContract]
    public class NodeInfo : IComparable<NodeInfo>
    {
        private const int M = 100;
        private static int _lowestAvailableId;
        
        [DataMember]
        public int Id;

        [DataMember]
        public EndpointAddress Address;

        public NodeInfo()
        {
            Id = _lowestAvailableId;
            ++_lowestAvailableId;
        }

        public int Distance(NodeInfo from)  //d(from.Id, Id)
        {
            return Math.Abs((Id - from.Id)%M);  //modulo?
        }

        public static int Distance(NodeInfo from, NodeInfo to)
        {
            return Math.Abs((to.Id - from.Id)%M);
        }

        public static bool operator <(NodeInfo x, NodeInfo y)
        {
            return x.Id < y.Id;  //modulo?
        }

        public static bool operator >(NodeInfo x, NodeInfo y)
        {
            return x.Id > y.Id;  //modulo?
        }

        public static int operator -(NodeInfo x, NodeInfo y)
        {
            return x.Id - y.Id;  //modulo?
        }

        public NodeInfo(int id, EndpointAddress address)
        {
            if (id < 0 || id >= M)
                throw new ArgumentOutOfRangeException(String.Format("Id should be a value between 0 and {0}",M-1));
            Id = id;
            Address = address;
        }

        public int CompareTo(NodeInfo obj)
        {
            return Id - obj.Id;
        }
    }
}
