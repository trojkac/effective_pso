using System;
using System.Collections.Generic;
using System.IO;
using Common;
using Node;

namespace UserInterface
{
    class Program
    {
        static void Main(string[] args)
        {
            UserNodeParameters nodeParams = ReadNodeParameters();
            UserFunctionParameters functionParams = ReadFunctionParameters();
            UserPsoParameters psoParams = ReadPsoParameters();

            MachineManager machineManager = new MachineManager(nodeParams, functionParams);
        }

        public static UserNodeParameters ReadNodeParameters()
        {
            Console.WriteLine("Podaj ścieżkę do pliku z parametrami węzła");
            string nodePath = Console.ReadLine();

            UserNodeParameters nodeParams = new UserNodeParameters();
            if (!ParametersReader.ReadNodeParametersFile(nodeParams, nodePath))
            {
                Console.WriteLine("Nie udało się wczytać pliku z danymi węzła");
            }
            return nodeParams;
        }

        public static UserFunctionParameters ReadFunctionParameters()
        {
            Console.WriteLine("Podaj ścieżkę do pliku z parametrami funkcji");
            string functionPath = Console.ReadLine();

            UserFunctionParameters functionParams = new UserFunctionParameters();
            if (!ParametersReader.ReadFunctionParametersFile(functionParams, functionPath))
            {
                Console.WriteLine("Nie udało się wczytać pliku z danymi funkcji");
            }
            return functionParams;
        }

        public static UserPsoParameters ReadPsoParameters()
        {
            Console.WriteLine("Podaj ścieżkę do pliku z parametrami PSO");
            string psoPath = Console.ReadLine();

            UserPsoParameters psoParams = new UserPsoParameters();
            if (!ParametersReader.ReadPsoParametersFile(psoParams, psoPath))
            {
                Console.WriteLine("Nie udało się wczytać pliku z danymi PSO");
            }
            return psoParams;
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

            public static bool ReadNodeParametersFile(UserNodeParameters parameters, string path, bool relativePath = true)
            {
                string[] lines = File.ReadAllLines(relativePath ? GetAbsolutePath(path) : path);
                string vcpus = lines[0];
                string isgpu = lines[1];


                int nrOfVCpu;
                if (!Int32.TryParse(vcpus, out nrOfVCpu))
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

                List<string> addresses = new List<string>();
                if (lines.Length > 2)
                {
                    string[] peers = lines[2].Split(',');
                    try
                    {
                        for (int i = 0; i < peers.Length; i++)
                        {
                            if (CheckIfValidNodeAddress(peers[i]))
                            {
                                addresses.Add(peers[i]);
                            }
                            else
                            {
                                Console.WriteLine("Napotkano niepoprawny adres IP");
                            }
                        }
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine("Błąd przy wczytywaniu adresów");
                        return false;
                    }
                }


                parameters.NrOfVCpu = nrOfVCpu;
                parameters.IsGpu = isGpu;
                parameters.PeerAddresses = addresses;

                return true;
            }

            public static bool ReadFunctionParametersFile(UserFunctionParameters parameters, string path, bool relativePath = true)
            {
                string[] lines = File.ReadAllLines(relativePath ? GetAbsolutePath(path) : path);
                string functionType = lines[0];
                string dimension = lines[1];
                string[] coefficients = lines[2].Split(',');
                string[] searchSpace = lines[3].Split(',');


                FitnessFunctionType ftype;
                if (!Enum.TryParse<FitnessFunctionType>(functionType, true, out ftype))
                {
                    Console.WriteLine("Nieznana funkcja");
                    return false;
                }

                int dim;
                if (!Int32.TryParse(dimension, out dim))
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

                parameters.FitnessFunctionType = ftype;
                parameters.Dimension = dim;
                parameters.Coefficients = coeff;
                parameters.SearchSpace = sspace;

                return true;
            }

            public static bool ReadPsoParametersFile(UserPsoParameters parameters, string path, bool relativePath = true)
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

                parameters.FullyInformedParticles = fi;
                parameters.StandardParticles = std;
                parameters.IterationsLimitCondition = isIterations;
                parameters.Iterations = iters;
                parameters.Epsilon = eps;
                parameters.TargetValueCondition = isTargetValue;
                parameters.TargetValue = target;

                return true;
            }
        }
    }
}
