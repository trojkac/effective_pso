using System;
using System.Collections.Generic;
using System.Net;
using System.Runtime.Serialization;

namespace NetworkManager
{
    [DataContract]
    public class NetworkNodeInfo : IComparable<NetworkNodeInfo>
    {
        [DataMember]
        private const ulong M = UInt64.MaxValue;

        [DataMember]
        public ulong Id
        {
            get
            {
                string[] parts = (TcpAddress.Split('/'))[2].Split(':');

                Byte[] bytes = (IPAddress.Parse(parts[0])).GetAddressBytes();

                ulong ip = (ulong)(BitConverter.ToInt32(bytes, 0));
                ulong port = (ulong)(Int32.Parse(parts[1]));

                return (ip << 32) + port;
            }
        }

        [DataMember]
        public string TcpAddress;

        [DataMember]
        public string PipeAddress;

        public NetworkNodeInfo()
        {
        }

        public NetworkNodeInfo(string tcpAddress, string pipeAddress)
        {
            TcpAddress = tcpAddress;
            PipeAddress = pipeAddress;
        }

        //ile id trzeba przejść, aby z from.Id dojść do to.Id idąc tylko w prawo
        public static ulong Distance(NetworkNodeInfo from, NetworkNodeInfo to)
        {
            if (to.Id > from.Id)
            {
                return to.Id - from.Id;
            }
            return M - (@from.Id - to.Id);
        }

        //returns networkNodeInfo with id closest (from the left) to to.Id
        public static NetworkNodeInfo GetClosestPeer(NetworkNodeInfo to, ICollection<NetworkNodeInfo> infos)
        {
            ulong minDistance = UInt64.MaxValue;
            NetworkNodeInfo closestPeer = null;

            foreach (NetworkNodeInfo nodeInfo in infos)
            {
                if (Distance(nodeInfo, to) < minDistance)
                {
                    minDistance = Distance(nodeInfo, to);
                    closestPeer = nodeInfo;
                }
            }

            return closestPeer;
        }

        //returns NetworkNodeInfo with id closest (from the right) to to.Id
        public static NetworkNodeInfo GetBestSuccessorCandidate(NetworkNodeInfo to, ICollection<NetworkNodeInfo> infos)
        {
            ulong minDistance = UInt64.MaxValue;
            NetworkNodeInfo closestPeer = null;

            foreach (NetworkNodeInfo nodeInfo in infos)
            {
                if (Distance(to, nodeInfo) < minDistance)
                {
                    minDistance = Distance(to, nodeInfo);
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

        public bool Equals(NetworkNodeInfo other)
        {
            return TcpAddress.Equals(other.TcpAddress) && PipeAddress.Equals(other.PipeAddress);  //or by Id
        }

        public int CompareTo(NetworkNodeInfo other)
        {
            throw new NotImplementedException();
        }

        public override int GetHashCode()
        {
            return (TcpAddress + PipeAddress).GetHashCode(); //TODO: maybe change it?
        }
    }
}