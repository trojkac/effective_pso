using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Runtime.Serialization;
using System.ServiceModel;
using System.Text;
using System.Threading.Tasks;

namespace Node
{
    [DataContract]
    public class NodeInfo : IComparable<NodeInfo>
    {
        private const int M = 100;
        
        [DataMember]
        public int Id;

        [DataMember]
        public EndpointAddress Address;

        public int Distance(NodeInfo other)
        {
            return Math.Abs(Id - other.Id)%M;
        }

        public static bool operator <(NodeInfo x, NodeInfo y)
        {
            return x.Id < y.Id;
        }

        public static bool operator >(NodeInfo x, NodeInfo y)
        {
            return x.Id > y.Id;
        }

        public static int operator -(NodeInfo x, NodeInfo y)
        {
            return x.Id - y.Id;
        }

        public NodeInfo(int id, EndpointAddress address)
        {
            if (id < 0 || id >= M)
                throw new ArgumentOutOfRangeException(String.Format("Id should be value between 0 and {0}",M-1));
            Id = id;
            Address = address;
        }

        public int CompareTo(NodeInfo obj)
        {
            return Id - obj.Id;
        }
    }
}
