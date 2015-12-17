// CppUtils.h

#pragma once

#include <algorithm>
#include <functional>
#include <vector>
#include <random>

using namespace System;

namespace CppUtils {

	public ref class Random
	{
    public:
        double Random::random_double()
        {
            return Random::random_in_range(0.0, 1.0);
        }

        double Random::random_in_range(double min, double max)
        {
            std::uniform_real_distribution<float> distribution(min, max);
            std::random_device rd;
            std::default_random_engine e(rd());

            return distribution(e);
        }

        std::vector<double> Random::random_vector(double len, double min, double max)
        {
            auto result = std::vector<double>();

            std::uniform_real_distribution<float> distribution(min, max);
            std::mt19937 engine;
            auto generator = bind(distribution, engine);
            generate(result.begin(), result.end(), generator);

            return result;
        }
	};
}
