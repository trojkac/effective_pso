﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CocoWrapper;

namespace ExampleExperiment
{
    class Program
    {

        /**
         * The maximal budget for evaluations done by an optimization algorithm equals 
         * dimension * BUDGET_MULTIPLIER.
         * Increase the budget multiplier value gradually to see how it affects the runtime.
         */
        public const int BUDGET_MULTIPLIER = 2;

        /**
         * The maximal number of independent restarts allowed for an algorithm that restarts itself. 
         */
        public const int INDEPENDENT_RESTARTS = 10000;

        /**
         * The random seed. Change if needed.
         */
        public const int RANDOM_SEED = 12;

        /**
         * The problem to be optimized (needed in order to simplify the interface between the optimization
         * algorithm and the COCO platform).
         */
        public static Problem PROBLEM;

        /**
        * Evaluate the static PROBLEM.
        */
        public static double[] evaluateFunction(double[] x)
        {
            return PROBLEM.evaluateFunction(x);
        }

        /**
        * Interface for function evaluation.
        */
        public delegate double[] OptimizationFunction(double[] x);  //można z tego zrobić obiekt OptFunc, jeżeli będą problemy

        public OptimizationFunction of = evaluateFunction;


        static void Main(string[] args)
        {
            Random randomGenerator = new Random(RANDOM_SEED);

            CocoLibraryWrapper.cocoSetLogLevel("info");

            Console.WriteLine("Running the example experiment... (might take time, be patient)");
            exampleExperiment("bbob", "bbob", randomGenerator);

            Console.WriteLine("Done");

            return;
        }

        public static void exampleExperiment(String suiteName, String observerName, Random randomGenerator)
        {
            try
            {

                /* Set some options for the observer. See documentation for other options. */
                String observerOptions =
                          "result_folder: RS_on_" + suiteName + " "
                        + "algorithm_name: RS "
                        + "algorithm_info: \"A simple random search algorithm\"";

                /* Initialize the suite and observer */
                Suite suite = new Suite(suiteName, "year: 2016", "dimensions: 2,3,5,10,20,40");
                Observer observer = new Observer(observerName, observerOptions);
                Benchmark benchmark = new Benchmark(suite, observer);

                /* Iterate over all problems in the suite */
                while ((PROBLEM = benchmark.getNextProblem()) != null)
                {

                    int dimension = PROBLEM.getDimension();

                    /* Run the algorithm at least once */
                    for (int run = 1; run <= 1; run++)
                    //for (int run = 1; run <= 1 + INDEPENDENT_RESTARTS; run++)
                    {

                        long evaluationsDone = PROBLEM.getEvaluations();
                        long evaluationsRemaining = (long)(dimension * BUDGET_MULTIPLIER) - evaluationsDone;

                        /* Break the loop if the target was hit or there are no more remaining evaluations */
                        if (PROBLEM.isFinalTargetHit() || (evaluationsRemaining <= 0))
                            break;

                        /* Call the optimization algorithm for the remaining number of evaluations */
                        myRandomSearch(evaluateFunction,
                                       dimension,
                                       PROBLEM.getNumberOfObjectives(),
                                       PROBLEM.getSmallestValuesOfInterest(),
                                       PROBLEM.getLargestValuesOfInterest(),
                                       evaluationsRemaining,
                                       randomGenerator);

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
        }

        /** 
	     * A simple random search algorithm that can be used for single- as well as multi-objective 
	     * optimization.
	     */
        public static void myRandomSearch(OptimizationFunction f,
                                          int dimension,
                                          int numberOfObjectives,
                                          double[] lowerBounds,
                                          double[] upperBounds,
                                          long maxBudget,
                                          Random randomGenerator)
        {

            double[] x = new double[dimension];
            double[] y = new double[numberOfObjectives];
            double range;

            for (int i = 0; i < maxBudget; i++)
            {

                /* Construct x as a random point between the lower and upper bounds */
                for (int j = 0; j < dimension; j++)
                {
                    range = upperBounds[j] - lowerBounds[j];
                    x[j] = lowerBounds[j] + randomGenerator.NextDouble() * range;
                }

                /* Call the evaluate function to evaluate x on the current problem (this is where all the COCO logging
                 * is performed) */
                y = f(x);
            }

        }
    }
}
