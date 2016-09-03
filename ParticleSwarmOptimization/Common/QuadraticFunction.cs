using System.Linq;
using Common.Parameters;

namespace Common
{
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
}