using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Common;

namespace Tests_Common
{
    public class PsoSettingsFactory
    {
        public static PsoParameters QuadraticFunction1DFrom3To5()
        {
            return new PsoParameters(new[] { new ParticlesCount(PsoParticleType.Standard, 40) }, new FunctionParameters()
                {
                    Dimension = 1,
                    Coefficients = new[] { 1.0 },
                    FitnessFunctionType = "quadratic",
                    SearchSpace = new[] { new DimensionBound(3, 5), }

                })
            {
                Epsilon = 0,
                Iterations = 1,
                IterationsLimitCondition = true,
                TargetValueCondition = false,
            };
        }
        public static PsoParameters QuadraticFunction20D()
        {
            var dim = 20;
            var settings = QuadraticFunction1DFrom3To5();
            settings.FunctionParameters.Dimension = dim;
            settings.Iterations = 1000;
            settings.IterationsLimitCondition = true;
            settings.FunctionParameters.SearchSpace = new DimensionBound[dim];
            settings.FunctionParameters.Coefficients = new double[dim];
            for (int i = 0; i < dim; i++)
            {
                settings.FunctionParameters.SearchSpace[i] = new DimensionBound(-4.0, 4.0);
                settings.FunctionParameters.Coefficients[i] = 1;
            }
            return settings;
        }
    }
}
