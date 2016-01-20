using System;
using System.Runtime.Serialization;

namespace Common
{
    [DataContract]
    public abstract class AbstractFitnessFunction
    {
        public AbstractFitnessFunction(UserFunctionParameters functionParams)
        {
            Dimension = functionParams.Dimension;
            Coefficients = new double[Dimension];
            functionParams.Coefficients.CopyTo(Coefficients, 0);
        }
        public static AbstractFitnessFunction GetFitnessFunction(UserFunctionParameters parameters)
        {
            switch (parameters.FitnessFunctionType)
            {
                case FitnessFunctionType.Quadratic:
                    return new QuadraticFunction(parameters);
                case FitnessFunctionType.Rastrigin:
                    return new RastriginFunction(parameters);
                case FitnessFunctionType.Rosenbrock:
                    return new RosenbrockFunction(parameters);
                default:
                    throw new ArgumentException("Unknown function type.");
            }
        }

        public abstract double Calculate(double[] vector);
        [DataMember]
        public int Dimension;
        [DataMember]
        public double[] Coefficients;
    }

    public class QuadraticFunction : AbstractFitnessFunction
    {


        public override double Calculate(double[] vector)
        {
            double value = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                value += Coefficients[i] * vector[i] * vector[i];
            }
            return value;
        }

        public QuadraticFunction(UserFunctionParameters functionParams)
            : base(functionParams)
        {

        }
    }

    public class RastriginFunction : AbstractFitnessFunction
    {

        public override double Calculate(double[] vector)
        {
            double value = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                value += vector[i] * vector[i] + Math.Cos(2 * Math.PI * vector[i]);
            }
            return value;
        }

        public RastriginFunction(UserFunctionParameters functionParams)
            : base(functionParams)
        {
        }
    }

    public class RosenbrockFunction : AbstractFitnessFunction
    {

        public override double Calculate(double[] vector)
        {
            double value = 0;
            for (int i = 0; i < vector.Length-1; i++)
            {
                value += ((1 - vector[i])*(1 - vector[i]) + 100 * (vector[i + 1] - vector[i] * vector[i]) *(vector[i + 1] - vector[i] * vector[i]));
            }
            return value;
        }

        public RosenbrockFunction(UserFunctionParameters functionParams)
            : base(functionParams)
        {
        }
    }
}
