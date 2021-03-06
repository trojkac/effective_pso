﻿using System;
using System.Collections.Generic;
using System.Linq;
using CocoWrapper;
using Common;
using Common.Parameters;
using Node;
using Narkhedegs.PerformanceMeasurement;
using ManagedGPU;

namespace CocoClusterApp
{
    class Program
    {

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
              try
              {
                machineManager.Register(nodeParams.PeerAddress);

              }
              catch (Exception e)
              {
                Console.WriteLine("Unexpected error occured. Plase try to connect once again.");
                return;
              }
                Console.WriteLine("Working...");
                Console.WriteLine("Press ENTER to finish");
                ConsoleKeyInfo pressed = new ConsoleKeyInfo();
                while (pressed.Key != ConsoleKey.Enter)
                {
                    pressed = Console.ReadKey();
                };
            }
            else
            {
                if (args.Length < 3)
                {
                    Console.WriteLine("CocoClusterApp <Dim1[,Dim2,Dim3...]> <FunctionsFrom> <FunctionsTo> <Budget>");
                    return;
                }
                var dims = args[0];
                var functionsFrom = int.Parse(args[1]);
                var functionsTo = int.Parse(args[2]);
                var budgetMultiplier = int.Parse(args[3]);

                var randomGenerator = RandomGenerator.GetInstance(RandomSeed);
                CocoLibraryWrapper.cocoSetLogLevel("warning");

                var functionsToOptimize = new List<string>();
                for (var i = functionsFrom; i <= functionsTo; i++)
                {
                    functionsToOptimize.Add(string.Format("f{0:D3}", i));
                }
                Console.WriteLine("Press any key on the keyboard when ready...");
                Console.ReadKey();
                Console.WriteLine("Running the example experiment... (might take time, be patient)");
                try
                {

                    /* Set some options for the observer. See documentation for other options. */
                    var observerOptions =
                        "result_folder: " +
                        String.Format("{0}P_{1}G", psoParams.ParticleIterationsToRestart,
                            psoParams.PsoIterationsToRestart) 
                        + " algorithm_name: PSO"
                        + " algorithm_info: \"A simple Random search algorithm\"";
                    /* Initialize the suite and observer */
                    var suite = new Suite("bbob", "year: 2016", "dimensions: " + dims);
                    var observer = new Observer("bbob", observerOptions);
                    var benchmark = new Benchmark(suite, observer);
                    /* Iterate over all problems in the suite */
                    var evalLogger = new EvaluationsLogger();
                    PsoServiceLocator.Instance.Register<EvaluationsLogger>(evalLogger);
                    var fileLogger = new FileLogger("evals.csv");
                    fileLogger.Log("function_id,gpu_evals,cpu_evals,gpu/cpu");
                    int instanceCounter = 0;
                    while ((Problem = benchmark.getNextProblem()) != null)
                    {
                        instanceCounter = (instanceCounter + 1) % 15;
                        if(instanceCounter == 0)
                        {
                            evalLogger.RestartCounters();
                        }
                        var restarts = -1;
                        FitnessFunction function;
                        if (!functionsToOptimize.Contains(Problem.FunctionNumber)) continue;
                        var settings = SetupOptimizer(psoParams, out function);
                        var evaluations = (long) settings.FunctionParameters.Dimension * budgetMultiplier;
                        var evaluationsLeft = evaluations;

                        do
                        {
                            restarts++;


                            settings.Iterations =
                                (int)Math.Ceiling(evaluations / ((double)settings.Particles.Sum(pc => pc.Count)));
                            //var sendParams = new PsoParameters()
                            //{
                            //    Iterations = psoParams.Iterations,
                            //    TargetValueCondition = psoParams.TargetValueCondition,
                            //    IterationsLimitCondition = psoParams.IterationsLimitCondition,
                            //    PsoIterationsToRestart = psoParams.PsoIterationsToRestart,
                            //    ParticleIterationsToRestart = psoParams.ParticleIterationsToRestart,
                            //    Epsilon = psoParams.Epsilon,
                            //    FunctionParameters = psoParams.FunctionParameters,
                            //    ParticlesCount = 20,
                            //    Particles = new ParticlesCount[1] { new ParticlesCount(PsoParticleType.Standard, 20)}
                            //};
                            machineManager.StartPsoAlgorithm(psoParams);
                            machineManager.GetResult();

                            var evalsDone = Problem.getEvaluations();
                            evaluationsLeft  = evaluations - evalsDone;
                        } while (!Problem.isFinalTargetHit() && evaluationsLeft > 0 );
                        evalLogger.IncreaseCpuEvals((int)Problem.getEvaluations());
                        Console.WriteLine("{0} | {1} evaluations | {2} restarts | {3:e} BestEval ", Problem.Id, Problem.getEvaluations(), restarts, function.BestEvaluation.FitnessValue[0]);
                        
                        if(instanceCounter == 14)
                        {
                            fileLogger.Log(String.Format("{0},{1},{2},{3}", Problem.Id.Substring(5,4), evalLogger._cpuEvaluations, evalLogger._gpuEvaluations, evalLogger.Ratio));
                        }



                    }
                    fileLogger.GenerateLog();
                    benchmark.finalizeBenchmark();
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                }


            }



            Console.WriteLine("Done");

        }

        private static PsoParameters SetupOptimizer(PsoParameters initialSettings, out FitnessFunction function)
        {
            var particlesNum = initialSettings.ParticlesCount;
            var settings = initialSettings;

            settings.TargetValueCondition = false;
            settings.IterationsLimitCondition = true;


            function = new FitnessFunction(Problem.evaluateFunction);
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
            return settings;

        }
    }
}
