using System;
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
      var nodeParams = nodeParamsDeserialize.Deserialize("nodeParams.xml");
      var machineManager = new MachineManager(nodeParams.Ip, nodeParams.Ports.ToArray());
      if (nodeParams.PeerAddress != null)
      {
        machineManager.Register(nodeParams.PeerAddress);

      }
      var cont = true;
      while (cont)
      {
        PrintMenu();
        var c = Console.ReadKey().KeyChar;
        Console.WriteLine("\n\n");

        switch (c)
        {
          case '1':
            var r = PerformCalculations(machineManager);
            Console.WriteLine("Value: {0}", r.FitnessValue[0]);
            break;
          case '0':
            cont = false;
            break;
        }
      }
    }

    private static void PrintMenu()
    {
      Console.WriteLine("1 - Start Calculations");
      Console.WriteLine("0 - Exit");
      Console.WriteLine("");
      Console.Write("choice:");

    }

    private static ParticleState PerformCalculations(MachineManager machineManager)
    {
      var psoParamsDeserialize = new ParametersSerializer<PsoParameters>();
      var psoParams = psoParamsDeserialize.Deserialize("psoParams.xml");

      machineManager.StartPsoAlgorithm(psoParams);
      return machineManager.GetResult();
    }
  }
}
