// PsoAlgorithm.hpp

#pragma once

#include "Particle.hpp"

#include "Function.hpp"


namespace ParticleSwarmOptimization {

	class PSOAlgorithm
	{
	public:
		PSOAlgorithm(
			Function *fitness_function,
			int iterations
			) :
			fitness_function_(fitness_function),
			max_iterations_(iterations),
			use_iterations_condition_(true),
			use_target_value_condition_(false)
		{}

		PSOAlgorithm(
			Function *fitness_function,
			double target_value,
			double delta
			) : fitness_function_(fitness_function), 
			max_iterations_(0),
			use_iterations_condition_(false),
			target_value_(target_value),
			use_target_value_condition_(true),
			delta_(delta) {}

		PSOAlgorithm(
			Function  *fitness_function,
			int iterations,
			double target_value,
			double delta
			) : fitness_function_(fitness_function),
			max_iterations_(iterations), 
			use_iterations_condition_(true),
			target_value_(target_value), 
			delta_(delta),
			use_target_value_condition_(true)
			{}
		~PSOAlgorithm()
		{
			delete(fitness_function_);
		}
		std::tuple<std::vector<double>, double> run(std::vector<Particle*> particles)
		{
			auto iteration = 0;
			auto current_distance_to_target = std::numeric_limits<double>::infinity();
			for (auto i = 0; i < particles.size(); ++i)
			{
				particles[i]->update_personal_best(fitness_function_);
			}

			while (condition_check(current_distance_to_target,iteration))
			{
				for (auto i = 0; i < particles.size(); ++i)
				{
					particles[i]->translate();
					auto particle_best = particles[i]->update_personal_best(fitness_function_);
					if (use_target_value_condition_) 
						update_distance_to_target(particle_best, current_distance_to_target);
				}

				for (auto i = 0; i < particles.size(); ++i)
				{
					particles[i]->update_neighborhood(particles);
					particles[i]->update_velocity();
				}
			}

			auto global_best = std::make_tuple(std::vector<double>(), std::numeric_limits<double>::infinity());

			for (auto i = 0; i < particles.size(); ++i)
			{
				auto temp = std::get<1>(particles[i]->get_personal_best());

				if (std::get<1>(particles[i]->get_personal_best()) < std::get<1>(global_best))
					global_best = particles[i]->get_personal_best();
			}

			return global_best;
		}

	private:
		Function* fitness_function_;
		int max_iterations_;
		bool use_target_value_condition_;
		bool use_iterations_condition_;
		double target_value_;
		double delta_;

		inline bool condition_check(double distance_to_target, int &iteration)
		{
			return (!use_iterations_condition_ || iteration++ < max_iterations_) && (!use_target_value_condition_ || distance_to_target > delta_);
		}

		inline void update_distance_to_target(std::tuple<std::vector<double>,double> particle_best, double &current_distance_to_target)
		{
			auto particle_distance_to_target = abs(std::get<1>(particle_best) -target_value_);
			if (particle_distance_to_target < current_distance_to_target)
			{
				current_distance_to_target = particle_distance_to_target;
			}
		}

	};
}
