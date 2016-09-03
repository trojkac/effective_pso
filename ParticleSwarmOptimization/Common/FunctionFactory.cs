using System;
using System.Linq;
using CocoWrapper;
using Common.Parameters;

namespace Common
{

    public static class FunctionFactory
    {
        public static IFitnessFunction<double[],double[]> GetFitnessFunction(FunctionParameters parameters)
        {
            if (parameters.FitnessFunctionType.Contains("bbob"))
            {
                var functionId = parameters.FitnessFunctionType.Split('_')[1];
                var observerOptions =
                             "result_folder: PSO_on_bbob_" +
                             DateTime.Now.Hour.ToString() + DateTime.Now.Minute.ToString()
                           + " algorithm_name: PSO"
                           + " algorithm_info: \"Cluster node for optimizing BBOB functions\"";
                
                var suite = new Suite("bbob", "year: 2016", "dimensions: " + parameters.Dimension);
                var observer = new Observer("bbob", observerOptions);
                var benchmark = new Benchmark(suite, observer);
                Problem problem;
                /* Iterate over all problems in the suite */
                while ((problem = benchmark.getNextProblem()) != null)
                {
                    if (problem.FunctionNumber != functionId) continue;
                    var upper = problem.getLargestValuesOfInterest();
                    var bounds =
                        problem.getSmallestValuesOfInterest()
                            .Select((x, i) => new DimensionBound(x, upper[i]))
                            .ToArray();
                    parameters.SearchSpace = bounds;
                    return new FitnessFunction(problem.evaluateFunction);
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
