using System;
using System.Linq;
using System.Runtime.Serialization;
using CocoWrapper;

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
    [DataContract]
    public abstract class AbstractFitnessFunction : IFitnessFunction<double[],double[]>
    {
        private IOptimization<double[]> _optimization; 
        protected AbstractFitnessFunction(FunctionParameters functionParams)
        {
            Dimension = functionParams.Dimension;
            Coefficients = new double[Dimension];
            functionParams.Coefficients.CopyTo(Coefficients, 0);
            _optimization = PsoServiceLocator.Instance.GetService<IOptimization<double[]>>();
        }

        public abstract double[] Calculate(double[] vector);
        [DataMember]
        public int Dimension;
        [DataMember]
        public double[] Coefficients;
        public double[] Evaluate(double[] x)
        {
            var state = new ParticleState(x,Calculate(x));
            if (BestEvaluation == null || _optimization.IsBetter(state.FitnessValue,BestEvaluation.FitnessValue) < 0)
            {
                BestEvaluation = state;
            }
            return state.FitnessValue;
        }

        public IState<double[], double[]> BestEvaluation { get; private set; }

        public int LocationDim
        {
            get { return Dimension; }
        }

        public int FitnessDim
        {
            get { return 1; }
        }
    }

    public class QuadraticFunction : AbstractFitnessFunction
    {


        public override double[] Calculate(double[] vector)
        {
            var value = vector.Select((x,i) => x*x*Coefficients[i]).Sum();
            return new []{value};
        }

        public QuadraticFunction(FunctionParameters functionParams)
            : base(functionParams)
        {

        }
    }

    public class RastriginFunction : AbstractFitnessFunction
    {

        public override double[] Calculate(double[] vector)
        {
            throw new NotImplementedException();
        }

        public RastriginFunction(FunctionParameters functionParams)
            : base(functionParams)
        {
        }
    }

    public class RosenbrockFunction : AbstractFitnessFunction
    {

        public override double[] Calculate(double[] vector)
        {
            throw new NotImplementedException();
        }

        public RosenbrockFunction(FunctionParameters functionParams)
            : base(functionParams)
        {
        }
    }
}
