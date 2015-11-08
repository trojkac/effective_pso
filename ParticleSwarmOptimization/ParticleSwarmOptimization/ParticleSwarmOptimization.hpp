// ParticleSwarmOptimization.h

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
			) : fitness_function_(fitness_function), max_iterations_(iterations) {}

		std::tuple<std::vector<double>, double> run(std::vector<Particle*> particles) const
		{
			int iteration = 0;

			for (int i = 0; i < particles.size(); ++i)
			{
				particles[i]->update_personal_best(fitness_function_);
			}

			while(iteration++ < max_iterations_)
			{
				for (int i = 0; i < particles.size(); ++i)
				{
					particles[i]->translate();
					auto particle_best = particles[i]->update_personal_best(fitness_function_);
				}

				for (int i = 0; i < particles.size(); ++i)
				{
					particles[i]->update_velocity(particles);
				}
			}

			std::tuple<std::vector<double>, double> global_best = std::make_tuple(std::vector<double>(), -std::numeric_limits<double>::infinity());

			for (int i = 0; i < particles.size(); ++i)
			{
				auto temp = std::get<1>(particles[i]->get_personal_best());

				if (std::get<1>(particles[i]->get_personal_best()) > std::get<1>(global_best))
					global_best = particles[i]->get_personal_best();
			}

			return global_best;
		}

	private:
		std::function<double(std::vector<double>)> fitness_function_;
		int max_iterations_;
	};
}
