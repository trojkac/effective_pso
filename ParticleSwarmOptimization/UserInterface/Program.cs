using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using Common;
using Microsoft.SqlServer.Server;

namespace UserInterface
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Podaj ścieżkę do pliku z parametrami PSO");
            string psoPath = Console.ReadLine();

            Console.WriteLine("Podaj ścieżkę do pliku z parametrami węzła");
            string nodePath = Console.ReadLine();

            RunParameters parameters = new RunParameters();
            ParametersReader reader = new ParametersReader();

            if (!reader.ReadNodeParametersFile(parameters, nodePath))
            {
                Console.WriteLine("Nie udało się wczytać pliku z danymi węzła");
            }

            if (!reader.ReadPsoParametersFile(parameters, psoPath))
            {
                Console.WriteLine("Nie udało się wczytać pliku z danymi PSO");
            }
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

            public bool ReadNodeParametersFile(RunParameters parameters, string path, bool relativePath = true)
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

            public bool ReadPsoParametersFile(RunParameters parameters, string path, bool relativePath = true)
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
        }
    }
}
