using System;
using System.Collections.Generic;
using System.Configuration;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Common;
using Node;

namespace UserInterface
{
    class Program
    {
        static void Main(string[] args)
        {
            NodeParameters nodeParams = ReadNodeParameters();
            FunctionParameters functionParams;

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
                        functionParams = ReadFunctionParameters();
                        machineManager.StartPsoAlgorithm(new PsoParameters(new Tuple<PsoParticleType, int>[]
                        {
                            new Tuple<PsoParticleType, int>(PsoParticleType.Standard, 15)
                        }, functionParams));
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



        public static NodeParameters ReadNodeParameters()
        {
            string nodePath = "n.txt";

            NodeParameters nodeParams = new NodeParameters();
            if (!ParametersReader.ReadNodeParametersFile(nodeParams, nodePath))
            {
                Console.WriteLine("Nie udało się wczytać pliku z danymi węzła");
            }
            return nodeParams;
        }

        public static FunctionParameters ReadFunctionParameters()
        {
            Console.WriteLine("Podaj ścieżkę do pliku z parametrami funkcji");
            string functionPath = Console.ReadLine();

            FunctionParameters functionParams = new FunctionParameters();
            if (!ParametersReader.ReadFunctionParametersFile(functionParams, functionPath))
            {
                Console.WriteLine("Nie udało się wczytać pliku z danymi funkcji");
            }
            return functionParams;
        }

        public class ParametersReader
        {
            public ParametersReader() { }

            public static string GetAbsolutePath(string filename)
            {
                return Path.Combine(Directory.GetParent(System.IO.Directory.GetCurrentDirectory()).Parent.Parent.FullName, filename);
            }

            public static bool CheckIfValidNodeAddress(string address)
            {
                return true;
            }

            public static bool ReadNodeParametersFile(NodeParameters parameters, string path, bool relativePath = true)
            {
                string[] lines = File.ReadAllLines(relativePath ? GetAbsolutePath(path) : path);
                string vcpus = lines[0];
                string isgpu = lines[1];
                string ports = lines[2];
                string ip = lines[3];
                string address = lines.Length > 4 ? lines[4] : null;



                int nrOfVCpu;
                if (!int.TryParse(vcpus, out nrOfVCpu))
                {
                    Console.WriteLine("Liczba VCpu powinna być liczbą całkowitą");
                    return false;
                }
                if (nrOfVCpu < 1)
                {
                    Console.WriteLine("Liczba VCpu powinna być większa od 0");
                    return false;
                }


                bool isGpu;
                if (!bool.TryParse(isgpu, out isGpu))
                {
                    Console.WriteLine("Błąd w IsGpu");
                    return false;
                }


                List<int> portList = new List<int>();
                string[] nrs = lines[2].Split(',');
                try
                {
                    for (int i = 0; i < nrs.Length; i++)
                    {
                        int p;
                        if (int.TryParse(nrs[i], out p))
                        {
                            portList.Add(p);
                        }
                        else
                        {
                            Console.WriteLine("Napotkano niepoprawny numer portu");
                        }
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine("Ports");
                    return false;
                }







                parameters.NrOfVCpu = nrOfVCpu;
                parameters.IsGpu = isGpu;
                parameters.Ports = portList;
                parameters.PeerAddress = address;
                parameters.Ip = ip;
                return true;
            }

            public static bool ReadFunctionParametersFile(FunctionParameters parameters, string path, bool relativePath = true)
            {
                string[] lines = File.ReadAllLines(relativePath ? GetAbsolutePath(path) : path);
                string functionType = lines[0];
                string dimension = lines[1];
                string[] coefficients = lines[2].Split(',');
                string[] searchSpace = lines[3].Split(',');

                int dim;
                if (!int.TryParse(dimension, out dim))
                {
                    Console.WriteLine("Liczba wymiarów powinna być liczbą całkowitą");
                    return false;
                }
                if (dim < 1)
                {
                    Console.WriteLine("Liczba wymiarów powinna być większa od 0");
                    return false;
                }

                double[] coeff = new double[dim];
                try
                {
                    for (int i = 0; i < dim; i++)
                    {
                        coeff[i] = double.Parse(coefficients[i]);
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine("Błąd we współczynnikach");
                    return false;
                }

                Tuple<double, double>[] sspace = new Tuple<double, double>[dim];
                try
                {
                    for (int i = 0; i < dim; i++)
                    {
                        sspace[i] = new Tuple<double, double>(double.Parse(searchSpace[2 * i]),
                            double.Parse(searchSpace[2 * i + 1]));
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine("Błąd w przestrzeni przeszukiwań");
                    return false;
                }

                parameters.FitnessFunctionType = functionType;
                parameters.Dimension = dim;
                parameters.Coefficients = coeff;
                parameters.SearchSpace = sspace;

                return true;
            }

            public static bool ReadPsoParametersFile(string path, bool relativePath = true)
            {
                string[] lines = File.ReadAllLines(relativePath ? GetAbsolutePath(path) : path);
                string iterationsLimitCondition = lines[0];
                string iterations = lines[1];
                string targetValueCondition = lines[2];
                string targetValue = lines[3];
                string epsilon = lines[4];
                string standard = lines[5];
                string fullyInformed = lines[6];

                bool isIterations;
                if (!bool.TryParse(iterationsLimitCondition, out isIterations))
                {
                    Console.WriteLine("iterationsLimitCondition");
                    return false;
                }

                int iters;
                if (!int.TryParse(iterations, out iters))
                {
                    Console.WriteLine("Condition");
                    return false;
                }

                bool isTargetValue;
                if (!bool.TryParse(targetValueCondition, out isTargetValue))
                {
                    Console.WriteLine("targetValueCondition");
                    return false;
                }

                double target;
                if (!double.TryParse(targetValue, out target))
                {
                    Console.WriteLine("targetValue");
                    return false;
                }

                double eps;
                if (!double.TryParse(epsilon, out eps))
                {
                    Console.WriteLine("Epsilon");
                    return false;
                }

                int std;
                if (!int.TryParse(standard, out std))
                {
                    Console.WriteLine("Standard particles");
                    return false;
                }

                int fi;
                if (!int.TryParse(fullyInformed, out fi))
                {
                    Console.WriteLine("Fully Informed Particles");
                    return false;
                }
                return true;
            }
        }
    }
}
