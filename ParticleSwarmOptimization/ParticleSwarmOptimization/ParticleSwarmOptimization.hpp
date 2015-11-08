// ParticleSwarmOptimization.h

#pragma once

#include "Particle.hpp"

#include <functional>

using namespace System;

namespace ParticleSwarmOptimization {

	public class PSOAlgorithm
	{
		PSOAlgorithm(
			std::function<double(std::vector<double>)> fitness_function, 
			int iterations
			) : fitness_function_(fitness_function), max_iterations_(iterations) {}

		std::tuple<std::vector<double>, double> run(std::vector<Particle> particles)
		{
			int iteration = 0;
			std::tuple<std::vector<double>, double> global_best;

			for (int i = 0; i < particles.size(); ++i)
			{
				particles[i].update_personal_best(fitness_function_);
			}

			while(iteration < max_iterations_)
			{
				for each (Particle particle in particles)
				{
					particle.translate();
					auto particle_best = particle.update_personal_best(fitness_function_);

					if (std::get<1>(particle_best) > std::get<1>(global_best))
						global_best = particle_best;
				}

				for each (Particle particle in particles)
				{
					particle.update_velocity(fitness_function_);
				}
			}
		}

	private:
		std::function<double(std::vector<double>)> fitness_function_;
		int max_iterations_;
	};
}
