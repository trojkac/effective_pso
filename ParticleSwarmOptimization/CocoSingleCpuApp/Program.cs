﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CocoWrapper;
using Common;
using Algorithm;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace CocoSingleCpuApp
{
    class Program
    {

        public const int BUDGET_MULTIPLIER = 100000;
        public const int INDEPENDENT_RESTARTS = 10000;
        public const int RANDOM_SEED = 12;
        public static Problem PROBLEM;
        static void Main(string[] args)
        {
            if (args.Length < 3)
            {
                Console.WriteLine("CocoSingleCpuApp <Dim1[,Dim2,Dim3...]> <FunctionsFrom> <FunctionsTo>");
                return;
            }
            string dims = args[0];

            var functionsFrom = int.Parse(args[1]);
            var functionsTo = int.Parse(args[2]);
            RandomGenerator randomGenerator = RandomGenerator.GetInstance(RANDOM_SEED);
            CocoLibraryWrapper.cocoSetLogLevel("info");
            var functionsToOptimize = new List<string>();
            for(var i = functionsFrom; i <= functionsTo; i++){
                functionsToOptimize.Add(String.Format("f{0:D3}",i));
            }
            Console.WriteLine("Running the example experiment... (might take time, be patient)");
            try
            {

                /* Set some options for the observer. See documentation for other options. */
                String observerOptions =
                          "result_folder: PSO_on_bbob_" +
                          DateTime.Now.Hour.ToString() + DateTime.Now.Minute.ToString()
                        + " algorithm_name: PSO"
                        + " algorithm_info: \"A simple Random search algorithm\"";
                /* Initialize the suite and observer */
                Suite suite = new Suite("bbob", "year: 2016", "dimensions: "+dims);
                Observer observer = new Observer("bbob", observerOptions);
                Benchmark benchmark = new Benchmark(suite, observer);
                /* Iterate over all problems in the suite */
                while ((PROBLEM = benchmark.getNextProblem()) != null)
                {
                    if (!functionsToOptimize.Contains(PROBLEM.FunctionNumber)) continue;
                    int dimension = PROBLEM.getDimension();
                    int particlesNum = dimension * 3;

                    /* Run the algorithm at least once */
                    //for (int run = 1; run <= 1; run++)
                    for (int run = 1; run <= 1 + INDEPENDENT_RESTARTS; run++)
                    {

                        long evaluationsDone = PROBLEM.getEvaluations();
                        long evaluationsRemaining = (long)(dimension * BUDGET_MULTIPLIER) - evaluationsDone;

                        /* Break the loop if the target was hit or there are no more remaining evaluations */
                        if (PROBLEM.isFinalTargetHit() || (evaluationsRemaining <= 0))
                            break;

                        var settings = new PsoSettings()
                        {
                            TargetValueCondition = false,
                            IterationsLimitCondition = true,
                            Iterations = (int)evaluationsRemaining,
                        };

                        var function = new FitnessFunction(PROBLEM.evaluateFunction);

                        var upper = PROBLEM.getLargestValuesOfInterest();
                        var bounds = PROBLEM.getSmallestValuesOfInterest().Select((x, i) => new Tuple<double, double>(x, upper[i])).ToArray();

                        function.FitnessDim = PROBLEM.getNumberOfObjectives();
                        function.LocationDim = PROBLEM.getDimension();

                        settings.FunctionParameters = new UserFunctionParameters { Dimension = function.LocationDim, SearchSpace = bounds };
                        settings.FunctionParameters.SearchSpace = bounds;
                        var particles = new IParticle[particlesNum];
                        for (var i = 0; i < particlesNum; i++)
                        {
                            particles[i] = ParticleFactory.Create(PsoParticleType.Standard, function.LocationDim,
                                function.FitnessDim, function, bounds);
                        }

                        PsoAlgorithm algorithm = new PsoAlgorithm(settings, function, particles);

                        algorithm.Run();
                        /* Call the optimization algorithm for the remaining number of evaluations */


                        /* Break the loop if the algorithm performed no evaluations or an unexpected thing happened */
                        if (PROBLEM.getEvaluations() == evaluationsDone)
                        {
                            Console.WriteLine("WARNING: Budget has not been exhausted (" + evaluationsDone + "/"
                                    + dimension * BUDGET_MULTIPLIER + " evaluations done)!\n");
                            break;
                        }
                        else if (PROBLEM.getEvaluations() < evaluationsDone)
                            Console.WriteLine("ERROR: Something unexpected happened - function evaluations were decreased!");
                    }

                }
                benchmark.finalizeBenchmark();
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
            Console.WriteLine("Done");

        }
    }
}