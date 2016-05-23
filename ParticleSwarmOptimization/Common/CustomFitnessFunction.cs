using System;
using System.Linq;
using System.Runtime.Serialization;

namespace Common
{
    [DataContract]
    public abstract class AbstractFitnessFunction : IFitnessFunction<double[],double[]>
    {
        private IOptimization<double[]> _optimization; 
        protected AbstractFitnessFunction(UserFunctionParameters functionParams)
        {
            Dimension = functionParams.Dimension;
            Coefficients = new double[Dimension];
            functionParams.Coefficients.CopyTo(Coefficients, 0);
            _optimization = PsoServiceLocator.Instance.GetService<IOptimization<double[]>>();
        }
        public static AbstractFitnessFunction GetFitnessFunction(UserFunctionParameters parameters)
        {
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

        public QuadraticFunction(UserFunctionParameters functionParams)
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

        public RastriginFunction(UserFunctionParameters functionParams)
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

        public RosenbrockFunction(UserFunctionParameters functionParams)
            : base(functionParams)
        {
        }
    }
}
