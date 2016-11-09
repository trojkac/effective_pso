using System;
using System.Collections.Generic;
using System.Linq;
using CocoWrapper;
using Common;
using Common.Parameters;
using Node;

namespace CocoClusterApp
{
    class Program
    {

        public const int RandomSeed = 12;
        public static Problem Problem;
        public static bool useCharged;
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
                while (pressed.Key != ConsoleKey.Enter)
                {
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
                var dims = args[0];
                var functionsFrom = int.Parse(args[1]);
                var functionsTo = int.Parse(args[2]);
                var budgetMultiplier = int.Parse(args[3]);
                useCharged = false;
                if (args.Length > 4)
                {
                    useCharged = bool.Parse(args[4]);
                }
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
                        String.Format("{0}P_{1}G{2}", psoParams.ParticleIterationsToRestart,
                            psoParams.PsoIterationsToRestart, useCharged ? "_charged" : "") 
                        + " algorithm_name: PSO"
                        + " algorithm_info: \"A simple Random search algorithm\"";
                    /* Initialize the suite and observer */
                    var suite = new Suite("bbob", "year: 2016", "dimensions: " + dims);
                    var observer = new Observer("bbob", observerOptions);
                    var benchmark = new Benchmark(suite, observer);
                    /* Iterate over all problems in the suite */
                    while ((Problem = benchmark.getNextProblem()) != null)
                    {
                        var restarts = -1;
                        FitnessFunction function;
                        if (!functionsToOptimize.Contains(Problem.FunctionNumber)) continue;
                        var evaluations = 0L;
                        var settings = SetupOptimizer(psoParams, out function);
                        var evauluations = settings.FunctionParameters.Dimension * budgetMultiplier;
                        var evaluationsLeft = evaluations;

                        do
                        {
                            restarts++;


                            settings.Iterations =
                                (int)Math.Ceiling(evaluations / ((double)settings.Particles.Sum(pc => pc.Count)));
                            var sendParams = new PsoParameters()
                            {
                                Iterations = psoParams.Iterations,
                                TargetValueCondition = psoParams.TargetValueCondition,
                                IterationsLimitCondition = psoParams.IterationsLimitCondition,
                                PsoIterationsToRestart = psoParams.PsoIterationsToRestart,
                                ParticleIterationsToRestart = psoParams.ParticleIterationsToRestart,
                                Epsilon = psoParams.Epsilon,
                                FunctionParameters = psoParams.FunctionParameters,
                                ParticlesCount = 20,
                                Particles = new ParticlesCount[1] { new ParticlesCount(PsoParticleType.Standard, 20)}
                            };
                            machineManager.StartPsoAlgorithm(psoParams, sendParams);
                            machineManager.GetResult();

                            var evalsDone = Problem.getEvaluations();
                            evaluationsLeft  = evaluations - evalsDone;
                        } while (!Problem.isFinalTargetHit() && evaluationsLeft > 0 );
                        Console.WriteLine("{0} | {1} evaluations | {2} restarts | {3:e} BestEval ", Problem.Id, Problem.getEvaluations(), restarts, function.BestEvaluation.FitnessValue[0]);




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
            settings.Particles = new[] { new ParticlesCount(PsoParticleType.DummyParticle, 1) };
            settings.ParticlesCount = 1;
            //settings.Particles = useCharged ?
            //    new[] { new ParticlesCount(PsoParticleType.Standard, 
            //        (int)Math.Ceiling(particlesNum/2.0)), 
            //        new ParticlesCount(PsoParticleType.ChargedParticle, (int)Math.Floor(particlesNum/2.0)),  }
            //        :
            //        new[] { new ParticlesCount(PsoParticleType.Standard, particlesNum) }

            //        ;
            return settings;

        }
    }
}
