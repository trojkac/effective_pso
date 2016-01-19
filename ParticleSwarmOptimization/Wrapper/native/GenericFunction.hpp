// ParticleSwarmOptimization.h

#pragma once

#include "../Stdafx.h"
#include "Function.hpp"
using namespace ParticleSwarmOptimization;

namespace ParticleSwarmOptimization {

	class GenericFunction : public Function{
	public:
		GenericFunction(std::function<double(std::vector<double>)> fitness_function, std::string name) :
			Function(name), best_value_(std::make_tuple(std::vector<double>(), std::numeric_limits<double>::infinity()))
		{
			fitness_function_ = fitness_function;
			fitness_function_pointer_ = NULL;
			name_ = name;
		}
		GenericFunction(std::function<double(std::vector<double>)> fitness_function) :
			best_value_(std::make_tuple(std::vector<double>(), std::numeric_limits<double>::infinity()))
		{
			fitness_function_ = fitness_function;
			fitness_function_pointer_ = NULL;
		}
		GenericFunction(double(*fitness_function)(double*)) : best_value_(std::make_tuple(std::vector<double>(), std::numeric_limits<double>::infinity()))
		{
			fitness_function_pointer_ = fitness_function;

		}
		GenericFunction(double(*fitness_function)(double*), std::string name) :
			Function(name), best_value_(std::make_tuple(std::vector<double>(),std::numeric_limits<double>::infinity()))
		{
			fitness_function_pointer_ = fitness_function;

		}


		double evaluate(std::vector<double> X) override
		{
			auto value =  !fitness_function_pointer_ ?
				fitness_function_(X)
				:
				fitness_function_pointer_(&X[0])
				;
			if (value < std::get<1>(best_value_))
			{
				best_value_ = std::make_tuple(X,value);
			}
			return value;

		}

		std::tuple<std::vector<double>,double> best_evaluation() override
		{
			return best_value_;
		}
	private:
		std::function<double(std::vector<double>)> fitness_function_;
		double(*fitness_function_pointer_)(double*);
		std::tuple<std::vector<double>, double> best_value_;
	};
}
