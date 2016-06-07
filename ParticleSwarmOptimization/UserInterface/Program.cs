using System;
using System.Collections.Generic;
using System.Configuration;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Common;
using Node;
using System.Xml.Serialization;
using System.Xml;

namespace UserInterface
{
    class Program
    {
        static void Main(string[] args)
        {
            NodeParameters nodeParams = new NodeParameters() { Ip = "127.0.0.1", IsGpu = true, NrOfVCpu = 8, PeerAddress = "127.0.0.1"};
            var xmlSerializer = new XmlSerializer(typeof(NodeParameters));
            TextWriter writer = new StreamWriter("nodeParams.xml");
            xmlSerializer.Serialize(writer, nodeParams);
            writer.Close();
            var particles =  new [] {new ParticlesCount(PsoParticleType.Standard, 40)};
            var function = new PsoParameters()
            {
                Epsilon = 0,
                Iterations = 1,
                IterationsLimitCondition = true,
                TargetValueCondition = false,
                Particles = particles,
                FunctionParameters = new FunctionParameters()
                {
                    Dimension = 1,
                    Coefficients = new[] { 1.0 },
                    FitnessFunctionType = "quadratic",
                    SearchSpace = new[] { new DimensionBound(3, 5), }

                }
            };

            var psoXmlSerializer = new XmlSerializer(typeof(PsoParameters));
            var psoWriter = new StreamWriter("psoParams.xml");
            psoXmlSerializer.Serialize(psoWriter, function);
            
           
        }
    }
}
