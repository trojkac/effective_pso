// PsoAlgorithm.hpp

#pragma once

#include "Particle.hpp"

#include "Function.hpp"
#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <string>
#include <time.h>
#include <stdio.h>
#include <chrono>
#include <cstdarg>

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
		std::tuple<std::vector<double>, double> run(std::vector<Particle*> particles, std::string id)
		{
			std::ofstream file;

			auto t = std::time(nullptr);
			struct tm * now = std::localtime(&t);

			char filename[80];
			strftime(filename, 80, "%Y-%m-%d_%H%M%S_", now);

			//////strcpy(filename, "C:/Users/rosiakh/Source/Repos/effective_pso/ParticleSwarmOptimization/Tests/bin/Debug");
			///////strcat(filename, fmt("-%d-%d-%d@%d.%d.%d.txt",
			//////////	now->tm_hour, now->tm_min, now->tm_sec,
				/////////now->tm_mday, now->tm_mon + 1,
				/////////now->tm_year + 1900).c_str());
			//strcat(filename, id);

			//std::string dateTime = std::currentDateTime();

			file.open(filename + id);


			auto current_distance_to_target = std::numeric_limits<double>::infinity();
			for (auto i = 0; i < particles.size(); ++i)
			{
				particles[i]->update_personal_best(fitness_function_);
			}

			int log_iterations = 0;
			int log_interval = 10;

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

				//LOGUJ
				if (log_iterations % log_interval == 0)
				{
					file << std::endl << "iterations: " << log_iterations << std::endl;
					for (int i = 0; i < particles.size(); i++)
					{
						file << "prtcl: " << i << " best: ";
						file << std::get<1>(particles[i]->get_personal_best());
						file << " in location: [ ";
						auto location = std::get<0>(particles[i]->get_personal_best());
						for (std::vector<double>::const_iterator i = location.begin(); i != location.end(); ++i)
							file << *i << ' ';
						file << "]" << std::endl;
						file << std::endl;
					}
				}
				++log_iterations;
			}
			file.close();
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
			return abs(particle_best - target_value_);
		}

		std::string fmt(const std::string& fmt, ...) {
			int size = 200;
			std::string str;
			va_list ap;
			while (1) {
				str.resize(size);
				va_start(ap, fmt);
				int n = vsnprintf((char*)str.c_str(), size, fmt.c_str(), ap);
				va_end(ap);
				if (n > -1 && n < size) {
					str.resize(n);
					return str;
				}
				if (n > -1)
					size = n + 1;
				else
					size *= 2;
			}
			return str;
		}

	};
}
