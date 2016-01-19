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
			iteration_(0),
			fitness_function_(fitness_function),
			max_iterations_(iterations),
			use_iterations_condition_(true),
			use_target_value_condition_(false)
		{}

		PSOAlgorithm(
			Function *fitness_function,
			double target_value,
			double delta
			) :
			iteration_(0), 
			fitness_function_(fitness_function),
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
			) :
			iteration_(0),
			fitness_function_(fitness_function),
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
			auto current_distance_to_target = std::numeric_limits<double>::infinity();
			for (auto i = 0; i < particles.size(); ++i)
			{
				particles[i]->update_personal_best(fitness_function_);
			}

			while (condition_check())
			{
				for (auto i = 0; i < particles.size(); ++i)
				{
					particles[i]->translate();
					auto particle_best = particles[i]->update_personal_best(fitness_function_);
				}

				for (auto i = 0; i < particles.size(); ++i)
				{
					particles[i]->update_neighborhood(particles);
					particles[i]->update_velocity();
				}
			}			
			return fitness_function_->best_evaluation();
		}

	private:
		Function* fitness_function_;
		int max_iterations_;
		bool use_target_value_condition_;
		bool use_iterations_condition_;
		double target_value_;
		double delta_;
		int iteration_;

		inline bool condition_check()
		{
			return (!use_iterations_condition_ || iteration_++ < max_iterations_) && (!use_target_value_condition_ || distance_to_target(fitness_function_->best_evaluation()) > delta_);
		}

		inline double distance_to_target(std::tuple<std::vector<double>, double> particle_best)
		{
			return abs(std::get<1>(particle_best) -target_value_);
		}
		inline double distance_to_target(double particle_best)
		{
			return abs(particle_best-target_value_);
		}

	};
}
