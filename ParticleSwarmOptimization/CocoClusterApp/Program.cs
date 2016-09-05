﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CocoWrapper;
using Common;
using Common.Parameters;
using Algorithm;
using System.Xml;
using System.Xml.Serialization;
using System.IO;
using Node;
using Narkhedegs.PerformanceMeasurement;
using ManagedGPU;

namespace CocoClusterApp
{
    class Program
    {

        public const int IndependentRestarts = 1;
        public const int RandomSeed = 12;
        public static Problem Problem;

        static void Main(string[] args)
        {
            
            var timeStr = DateTime.Now.Hour.ToString("D2") + DateTime.Now.Minute.ToString("D2");

            var nodeParamsDeserialize = new ParametersSerializer<NodeParameters>();
            var psoParamsDeserialize = new ParametersSerializer<PsoParameters>();
            var nodeParams = nodeParamsDeserialize.Deserialize("nodeParams.xml");
            var psoParams = psoParamsDeserialize.Deserialize("psoParams.xml");

            MachineManager machineManager = new MachineManager(nodeParams.Ip, nodeParams.Ports.ToArray(), nodeParams.NrOfVCpu);
            if (nodeParams.PeerAddress != null)
            {
                machineManager.Register(nodeParams.PeerAddress);
                Console.WriteLine("Working...");
                Console.WriteLine("Press ENTER to finish");
                ConsoleKeyInfo pressed = new ConsoleKeyInfo();
                while(pressed.Key != ConsoleKey.Enter){
                    pressed = Console.ReadKey();
                };
            }
            else
            {
                if (args.Length < 3)
                {
                    Console.WriteLine("CocoSingleCpuApp <Dim1[,Dim2,Dim3...]> <FunctionsFrom> <FunctionsTo> <Budget>");
                    return;
                }
                var logger = new FileLogger("time_log_" + timeStr);
                var chronometer = new Chronometer();
                string dims = args[0];
                var functionsFrom = int.Parse(args[1]);
                var functionsTo = int.Parse(args[2]);
                var budgetMultiplier = int.Parse(args[3]);
                RandomGenerator randomGenerator = RandomGenerator.GetInstance(RandomSeed);
                CocoLibraryWrapper.cocoSetLogLevel("info");

                var functionsToOptimize = new List<string>();
                for (var i = functionsFrom; i <= functionsTo; i++)
                {
                    functionsToOptimize.Add(String.Format("f{0:D3}", i));
                }
                Console.WriteLine("Press any key on the keyboard when ready...");
                Console.ReadKey();
                Console.WriteLine("Running the example experiment... (might take time, be patient)");
                try
                {

                    /* Set some options for the observer. See documentation for other options. */
                    var observerOptions =
                        "result_folder: PSO_on_bbob_" +
                        timeStr
                        + " algorithm_name: PSO"
                        + " algorithm_info: \"A simple Random search algorithm\"";
                    /* Initialize the suite and observer */
                    var suite = new Suite("bbob", "year: 2016", "dimensions: " + dims);
                    var observer = new Observer("bbob", observerOptions);
                    var benchmark = new Benchmark(suite, observer);
                    /* Iterate over all problems in the suite */
                    while ((Problem = benchmark.getNextProblem()) != null)
                    {
                        if (!functionsToOptimize.Contains(Problem.FunctionNumber)) continue;

                        var dimension = Problem.getDimension();
                        var particlesNum = dimension * 3;
                        var evaluations = (long)(dimension * budgetMultiplier);

                        /* Break the loop if the target was hit or there are no more remaining evaluations */
                        if (Problem.isFinalTargetHit() || (evaluations <= 0))
                            break;

                        var settings = psoParams;
                        settings.TargetValueCondition = false;
                        settings.IterationsLimitCondition = true;
                        settings.Iterations = (int)Math.Ceiling(evaluations /(double)particlesNum);

                        var function = new FitnessFunction(Problem.evaluateFunction);
                        FunctionFactory.SaveToCache(Problem.Id, function);
                        var upper = Problem.getLargestValuesOfInterest();
                        var bounds =
                            Problem.getSmallestValuesOfInterest()
                                .Select((x, i) => new DimensionBound(x, upper[i]))
                                .ToArray();

                        function.FitnessDim = Problem.getNumberOfObjectives();
                        function.LocationDim = Problem.getDimension();
                        settings.FunctionParameters = new FunctionParameters
                        {
                            Dimension = function.LocationDim,
                            SearchSpace = bounds,
                            FitnessFunctionType = Problem.Id
                        };
                        settings.FunctionParameters.SearchSpace = bounds;
                        
                        settings.Particles = new ParticlesCount[] { new ParticlesCount(PsoParticleType.Standard, particlesNum) };
                        var runtimeMs = chronometer.Measure(() =>
                        {
                            machineManager.StartPsoAlgorithm(psoParams);
                            machineManager.GetResult();
                        });
                        logger.Log(String.Format("{0}: {1} {2}", Problem.Id, function.EvaluationsCount, runtimeMs));
                    }
                    benchmark.finalizeBenchmark();
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                }


            }



            Console.WriteLine("Done");

        }
    }
}
