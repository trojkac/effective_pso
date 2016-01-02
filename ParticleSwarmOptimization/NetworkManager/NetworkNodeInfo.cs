using System;
using System.Collections.Generic;
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

        public static int Distance(NetworkNodeInfo from, NetworkNodeInfo to)
        {
            return (to.Id + M - from.Id) % M;  // ile id trzeba przejść, aby z from.Id dojść do to.Id idąc tylko w prawo
        }

        //returns networkNodeInfo with id closest (from the left) to to.Id
        public static NetworkNodeInfo GetClosestPeer(NetworkNodeInfo to, ICollection<NetworkNodeInfo> infos)
        {
            int minDistance = Int32.MaxValue;
            NetworkNodeInfo closestPeer = null;

            foreach (NetworkNodeInfo nodeInfo in infos)
            {
                if (NetworkNodeInfo.Distance(nodeInfo, to) < minDistance)
                {
                    minDistance = NetworkNodeInfo.Distance(nodeInfo, to);
                    closestPeer = nodeInfo;
                }
            }

            return closestPeer;
        }

        //returns NetworkNodeInfo with id closest (from the right) to to.Id
        public static NetworkNodeInfo GetBestSuccessorCandidate(NetworkNodeInfo to, ICollection<NetworkNodeInfo> infos)
        {
            int minDistance = Int32.MaxValue;
            NetworkNodeInfo closestPeer = null;

            foreach (NetworkNodeInfo nodeInfo in infos)
            {
                if (NetworkNodeInfo.Distance(to, nodeInfo) < minDistance)
                {
                    minDistance = NetworkNodeInfo.Distance(to, nodeInfo);
                    closestPeer = nodeInfo;
                }
            }

            return closestPeer;
        }

        //returns true iff v is between u and w, i.e. v is strictly closer to u than w is and u != v (u < v < w)
        public static bool IsBetween(NetworkNodeInfo v, NetworkNodeInfo u, NetworkNodeInfo w)
        {
            return Distance(u, v) < Distance(u, w) && Distance(u, v) > 0;
        }

        public int CompareTo(NetworkNodeInfo obj)
        {
            return Id - obj.Id;
        }

        public bool Equals(NetworkNodeInfo other)
        {
            return TcpAddress.Equals(other.TcpAddress) && PipeAddress.Equals(other.PipeAddress);  //or by Id
        }

        public override int GetHashCode()
        {
            return (TcpAddress + PipeAddress).GetHashCode(); //TODO: maybe change it?
        }
    }
}