using System;
using System.Collections.Generic;
using System.Linq;
using CocoWrapper;
using Common.Parameters;

namespace Common
{

    public static class FunctionFactory
    {
        private static int cacheLimit = 5;
        private static Dictionary<string, IFitnessFunction<double[], double[]>>  functionCache = new Dictionary<string, IFitnessFunction<double[], double[]>>();

        public static void SaveToCache(string id, IFitnessFunction<double[], double[]> function)
        {
            if (functionCache.Count == cacheLimit)
            {
                functionCache.Clear();
            }
            functionCache.Add(id,function);
            if (functionCache.Count > cacheLimit)
            {
                functionCache.Clear();
            }
        }

        private static Benchmark benchmark = null;
        private static Observer observer = null;

        private static void restartBbob(int dimension)
        {
            var suite = new Suite("bbob", "year: 2016", "dimensions: " + dimension);
            if (observer == null)
            {
                var observerOptions =
                    "result_folder: PSO_on_bbob_" +
                    DateTime.Now.Hour.ToString() + DateTime.Now.Minute.ToString()
                    + " algorithm_name: PSO"
                    + " algorithm_info: \"Cluster node for optimizing BBOB functions\"";
                observer = new Observer("observer", observerOptions);
            }

            benchmark = new Benchmark(suite, observer);
        }

        public static IFitnessFunction<double[],double[]> GetFitnessFunction(FunctionParameters parameters)
        {
            if (functionCache.ContainsKey(parameters.FitnessFunctionType))
            {
                return functionCache[parameters.FitnessFunctionType];
            }
            if (parameters.FitnessFunctionType.Contains("bbob"))
            {
                var functionId = parameters.FitnessFunctionType;
                Problem problem;
                /* Iterate over all problems in the suite */
                restartBbob(parameters.Dimension);
                while ((problem = benchmark.getNextProblem()) != null)
                {
                    if (problem.Id != functionId) continue;
                    var upper = problem.getLargestValuesOfInterest();
                    var bounds =
                        problem.getSmallestValuesOfInterest()
                            .Select((x, i) => new DimensionBound(x, upper[i]))
                            .ToArray();
                    parameters.SearchSpace = bounds;
                    var function = new FitnessFunction(problem.evaluateFunction);
                    return function;
                }


            }
            switch (parameters.FitnessFunctionType)
            {
                case "quadratic":
                    return new QuadraticFunction(parameters);
                case "rastrigin":
                    return new RastriginFunction(parameters);
                case "rosenbrock":
                    return new RosenbrockFunction(parameters);
                default:
                    throw new ArgumentException("Unknown function type.");
            }
        }
    }
}
