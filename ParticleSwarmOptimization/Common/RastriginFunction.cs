using System;
using Common.Parameters;

namespace Common
{
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
}