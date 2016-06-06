﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Common;

namespace Tests
{
    public class PsoSettingsFactory
    {
        public static PsoParameters QuadraticFunction1DFrom3To5()
        {
            return new PsoParameters()
                {
                    Epsilon = 0,
                    Iterations = 1,
                    IterationsLimitCondition = true,
                    TargetValueCondition = false,
                    Particles = new[] {new Tuple<PsoParticleType, int>(PsoParticleType.Standard, 40)},
                    FunctionParameters = new FunctionParameters()
                    {
                        Dimension = 1,
                        Coefficients = new []{ 1.0 },
                        FitnessFunctionType = "quadratic",
                        SearchSpace = new []{new Tuple<double, double>(3,5), }

                    }
                };
        }
        public static PsoParameters QuadraticFunction20D()
        {
            var dim = 20;
            var settings = PsoSettingsFactory.QuadraticFunction1DFrom3To5();
            settings.FunctionParameters.Dimension = dim;
            settings.Iterations = 1000;
            settings.IterationsLimitCondition = true;
            settings.FunctionParameters.SearchSpace = new Tuple<double, double>[dim];
            settings.FunctionParameters.Coefficients = new double[dim];
            for (int i = 0; i < dim; i++)
            {
                settings.FunctionParameters.SearchSpace[i] = new Tuple<double, double>(-4.0, 4.0);
                settings.FunctionParameters.Coefficients[i] = 1;
            }
            return settings;
        }
    }
}
