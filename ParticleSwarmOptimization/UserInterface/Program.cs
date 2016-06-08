using System;
using System.Collections.Generic;
using System.Configuration;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Xml.Serialization;
using Common;
using Node;

namespace UserInterface
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var nodeParamsSerializer = new XmlSerializer(typeof (NodeParameters));
            var psoParamsSerializer = new XmlSerializer(typeof (PsoParameters));
            NodeParameters nodeParams;
            PsoParameters psoParams;
            using (var nodeFileReader = new StreamReader("nodeParams.xml"))
            {
                nodeParams = (NodeParameters)nodeParamsSerializer.Deserialize(nodeFileReader);    
            }
            using (var psoFileReader = new StreamReader("psoParams.xml"))
            {
                psoParams = (PsoParameters) psoParamsSerializer.Deserialize(psoFileReader);
            }
            MachineManager machineManager = new MachineManager(nodeParams.Ip, nodeParams.Ports.ToArray(), nodeParams.NrOfVCpu);
            if (nodeParams.PeerAddress != null)
            {
                machineManager.Register(nodeParams.PeerAddress);
            }

            char c = 'c';
            bool cont = true;
            while (cont)
            {
                Console.WriteLine("1 - Start Calculations");
                Console.WriteLine("0 - Exit");
                Console.WriteLine("");
                Console.Write("choice:");
                c = Console.ReadKey().KeyChar;
                Console.WriteLine("\n\n");

                switch (c)
                {
                    case '1':
                        machineManager.StartPsoAlgorithm(psoParams);
                        var r = machineManager.GetResult();
                        Console.WriteLine("Value: {0}",r.FitnessValue);
                        break;
                    case '0':
                        cont = false;
                        break;
                    default:
                        break;


                }
            }
        }
    }
}
