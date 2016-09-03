using System;
using System.Collections.Generic;
using System.Configuration;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Xml.Serialization;
using Common;
using Common.Parameters;
using Node;

namespace UserInterface
{
    public class Program
    {
        public static void Main(string[] args)
        {

            var nodeParamsDeserialize = new ParametersSerializer<NodeParameters>();
            var psoParamsDeserialize = new ParametersSerializer<PsoParameters>();
            var nodeParams = nodeParamsDeserialize.Deserialize("nodeParams.xml");
            var psoParams = psoParamsDeserialize.Deserialize("psoParams.xml");
            
            var machineManager = new MachineManager(nodeParams.Ip,nodeParams.Ports.ToArray());

            var cont = true;
            while (cont)
            {
                Console.WriteLine("1 - Start Calculations");
                Console.WriteLine("0 - Exit");
                Console.WriteLine("");
                Console.Write("choice:");
                var c = Console.ReadKey().KeyChar;
                Console.WriteLine("\n\n");

                switch (c)
                {
                    case '1':
                        machineManager.StartPsoAlgorithm(psoParams);
                        var r = machineManager.GetResult();
                        Console.WriteLine("Value: {0}",r.FitnessValue[0]);
                        break;
                    case '0':
                        cont = false;
                        break;
                }
            }
        }
    }
}
