using System;

namespace Common
{
    public abstract class AbstractFitnessFunction
    {
        public abstract double Calculate(double[] vector);
    }

    public class QuadraticFunction : AbstractFitnessFunction
    {
        public int Dimension;
        public double[] Coefficients;

        public override double Calculate(double[] vector)
        {
            double value = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                value += -Coefficients[i] * vector[i] * vector[i];
            }
            return value;
        }

        public QuadraticFunction(UserFunctionParameters functionParams)
        {
            Dimension = functionParams.Dimension;
            Coefficients = new double[Dimension];
            functionParams.Coefficients.CopyTo(Coefficients, 0);
        }
    }

    public class RastriginFunction : AbstractFitnessFunction
    {
        public int Dimension;
        public double[] Coefficients;

        public override double Calculate(double[] vector)
        {
            throw new NotImplementedException();
        }

        public RastriginFunction(UserFunctionParameters functionParams)
        {
            Dimension = functionParams.Dimension;
            Coefficients = new double[Dimension];
            functionParams.Coefficients.CopyTo(Coefficients, 0);
        }
    }

    public class RosenbrockFunction : AbstractFitnessFunction
    {
        public int Dimension;
        public double[] Coefficients;

        public override double Calculate(double[] vector)
        {
            throw new NotImplementedException();
        }

        public RosenbrockFunction(UserFunctionParameters functionParams)
        {
            Dimension = functionParams.Dimension;
            Coefficients = new double[Dimension];
            functionParams.Coefficients.CopyTo(Coefficients, 0);
        }
    }
}
