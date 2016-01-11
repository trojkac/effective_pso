// ParticleSwarmOptimization.h

#pragma once

#include "../Stdafx.h"
#include "Function.hpp"
using namespace ParticleSwarmOptimization;

namespace ParticleSwarmOptimization {

	class GenericFunction : public Function{
	public:
		GenericFunction(std::function<double(std::vector<double>)> fitness_function, std::string name) :
			Function(name)
		{
			fitness_function_ = fitness_function;
			fitness_function_pointer_ = NULL;
			name_ = name;
		}
		GenericFunction(std::function<double(std::vector<double>)> fitness_function)
		{
			fitness_function_ = fitness_function;
			fitness_function_pointer_ = NULL;
		}
		GenericFunction(double(*fitness_function)(double*))
		{
			fitness_function_pointer_ = fitness_function;

		}
		GenericFunction(double(*fitness_function)(double*), std::string name) :
			Function(name)
		{
			fitness_function_pointer_ = fitness_function;

		}


		double evaluate(std::vector<double> X) override
		{
			return !fitness_function_pointer_ ? 
				fitness_function_(X)
				:
				fitness_function_pointer_(&X[0])
				;

		}
	private:
		std::function<double(std::vector<double>)> fitness_function_;
		double(*fitness_function_pointer_)(double*);
	};
}
