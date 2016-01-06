// PsoAlgorithm.hpp

#pragma once

#include "Particle.hpp"
#include <functional>


namespace ParticleSwarmOptimization {

	class PSOAlgorithm
	{
	public:
		PSOAlgorithm(
			std::function<double(std::vector<double>)> fitness_function,
			int iterations
		);

		PSOAlgorithm(
			std::function<double(std::vector<double>)> fitness_function,
			double target_value,
			double delta
		);

		PSOAlgorithm(
			std::function<double(std::vector<double>)> fitness_function,
			int iterations,
			double target_value,
			double delta
		);

		std::tuple<std::vector<double>, double> run(std::vector<Particle*> particles) const;

	private:
		std::function<double(std::vector<double>)> fitness_function_;
		int max_iterations_;
		double target_value_;
		double delta_;

	};
}
