using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Common;

namespace Tests
{
    public class PsoSettingsFactory
    {
        public static PsoSettings QuadraticFunction1DFrom3To5()
        {
            return new PsoSettings()
                {
                    Dimensions = 1,
                    Epsilon = 0,
                    Iterations = 1,
                    IterationsLimitCondition = true,
                    TargetValueCondition = false,
                    Particles = new[] {new Tuple<PsoParticleType, int>(PsoParticleType.Standard, 10)},
                    FunctionParameters = new UserFunctionParameters()
                    {
                        Dimension = 1,
                        Coefficients = new []{ 1.0 },
                        FitnessFunctionType = FitnessFunctionType.Quadratic,
                        SearchSpace = new []{new Tuple<double, double>(3,5), }

                    }
                };
        }
    }
}
