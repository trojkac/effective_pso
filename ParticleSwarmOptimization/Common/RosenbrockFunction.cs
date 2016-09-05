using System;
using Common.Parameters;

namespace Common
{
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