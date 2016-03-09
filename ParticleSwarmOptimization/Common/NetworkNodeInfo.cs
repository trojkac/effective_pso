using System;
using System.Collections.Generic;
using System.Net;
using System.Runtime.Serialization;

namespace Common
{
    [DataContract]
    public class NetworkNodeInfo : IComparable<NetworkNodeInfo>
    {
        [DataMember]
        private const ulong M = UInt64.MaxValue;

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

        [DataMember] 
        public Uri[] ProxyParticlesAddresses;

        public NetworkNodeInfo()
        {
        }

        public NetworkNodeInfo(string tcpAddress, string pipeAddress)
        {
            TcpAddress = tcpAddress;
            PipeAddress = pipeAddress;
        }


        public int CompareTo(NetworkNodeInfo other)
        {
            return Id.CompareTo(other);
        }
    }
}