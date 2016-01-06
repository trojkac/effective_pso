#include "PsoAlgorithm.hpp"

ParticleSwarmOptimization::PSOAlgorithm::PSOAlgorithm(std::function<double(std::vector<double>)> fitness_function, int iterations): fitness_function_(fitness_function), max_iterations_(iterations)
{}

ParticleSwarmOptimization::PSOAlgorithm::PSOAlgorithm(std::function<double(std::vector<double>)> fitness_function, double target_value, double delta): fitness_function_(fitness_function), max_iterations_(0), target_value_(target_value), delta_(delta)
{}

ParticleSwarmOptimization::PSOAlgorithm::PSOAlgorithm(std::function<double(std::vector<double>)> fitness_function, int iterations, double target_value, double delta): fitness_function_(fitness_function), max_iterations_(iterations), target_value_(target_value), delta_(delta)
{}

std::tuple<std::vector<double>, double> ParticleSwarmOptimization::PSOAlgorithm::run(std::vector<Particle*> particles) const
{
	auto iteration = 0;
	auto current_distance_to_target = std::numeric_limits<double>::infinity();
	auto use_target_value_condition = target_value_ && delta_;
	auto use_iterations_condition = max_iterations_ > 0;
	for (auto i = 0; i < particles.size(); ++i)
	{
		particles[i]->update_personal_best(fitness_function_);
	}

	while ((!use_iterations_condition || iteration++ < max_iterations_) && (!use_target_value_condition || current_distance_to_target > delta_))
	{
		for (auto i = 0; i < particles.size(); ++i)
		{
			particles[i]->translate();
			auto particle_best = particles[i]->update_personal_best(fitness_function_);
			if (use_target_value_condition)
			{
				auto particle_distance_to_target = abs(std::get<1>(particle_best) - target_value_);
				if (particle_distance_to_target < current_distance_to_target)
				{
					current_distance_to_target = particle_distance_to_target;
				}
			}
		}

		for (auto i = 0; i < particles.size(); ++i)
		{
			particles[i]->update_neighborhood(particles);
			particles[i]->update_velocity();
		}
	}

	auto global_best = std::make_tuple(std::vector<double>(), -std::numeric_limits<double>::infinity());

	for (auto i = 0; i < particles.size(); ++i)
	{
		auto temp = std::get<1>(particles[i]->get_personal_best());

		if (std::get<1>(particles[i]->get_personal_best()) > std::get<1>(global_best))
			global_best = particles[i]->get_personal_best();
	}

	return global_best;
}
