// ParticleSwarmOptimization.h

#pragma once

#include "../Stdafx.h"
#include "Particle.hpp"
using namespace ParticleSwarmOptimization;

namespace ParticleSwarmOptimization {

	class StandardParticle : public Particle {
	public:
	    explicit StandardParticle(int dimensions) : dimensions_(dimensions)
		{
			personal_best_ = make_tuple(std::vector<double>(2), -std::numeric_limits<double>::infinity());
		}

		void init_location() override
		{
			std::vector<double> location(dimensions_);
			// this probably should go to static field or we should create ParticlesFactory responsible 
			// for generating particles with proper distribution of location and in specific limits
			std::uniform_real_distribution<float> distribution(0.0f, 10.0f); 
			std::mt19937 engine; // Mersenne twister MT19937
			auto generator = bind(distribution, engine);
			generate(location.begin(), location.end(), generator); 
			// to remove after improving generation logic
			if (bounds_.size() == dimensions_) {
				transform(location.begin(), location.end(), bounds_.begin(), location.begin(), &clamp);
			}
			location_ = location;
		}

		void init_velocity() override
		{
			std::vector<double> velocity(dimensions_);
			std::uniform_real_distribution<float> distribution(0.0f, 2.0f); //Values between 0 and 2
			std::mt19937 engine; // Mersenne twister MT19937
			auto generator = std::bind(distribution, engine);
			generate(velocity.begin(), velocity.end(), generator);
			velocity_ = velocity;
		}

		std::tuple<std::vector<double>, double> update_personal_best(std::function<double(std::vector<double>)> function) override
		{
			auto value = function(location_);
			return std::get<1>(personal_best_) < value ?
				(personal_best_ = make_tuple(location_, value)) : personal_best_;
		}

		void update_neighborhood(std::vector<Particle*> all_particles) override
		{
			if (neighborhood_.size() <= 0 )
			{
				neighborhood_ = all_particles;
			}
		}

		void update_velocity() override
		{
			std::tuple<std::vector<double>, double> global_best = personal_best_;

			for (int i = 0; i < neighborhood_.size(); ++i)
			{
				if (std::get<1>(neighborhood_[i]->get_personal_best()) > std::get<1>(global_best))
					global_best = neighborhood_[i]->get_personal_best();
			}

			std::vector<double> to_personal_best(dimensions_);
			std::vector<double> to_global_best(dimensions_);

			transform(std::get<0>(personal_best_).begin(), std::get<0>(personal_best_).end(), 
				location_.begin(), to_personal_best.begin(), std::minus<double>());

			transform(std::get<0>(global_best).begin(), std::get<0>(global_best).end(), location_.begin(), to_global_best.begin(),
				std::minus<double>());

			std::random_device rd;
			std::default_random_engine e1(rd());
			std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
			double phi1 = uniform_dist(e1);
			double phi2 = uniform_dist(e1);

			transform(to_personal_best.begin(), to_personal_best.end(), to_personal_best.begin(),
				bind1st(std::multiplies<double>(), phi1));

			transform(to_global_best.begin(), to_global_best.end(), to_global_best.begin(),
				bind1st(std::multiplies<double>(), phi2));

			transform(velocity_.begin(), velocity_.end(), to_global_best.begin(), velocity_.begin(), std::plus<double>());
			transform(velocity_.begin(), velocity_.end(), to_personal_best.begin(), velocity_.begin(), std::plus<double>());
		}

		void translate() override
		{
			transform(location_.begin(), location_.end(), velocity_.begin(),
				location_.begin(), std::plus<double>());
			if (bounds_.size() == dimensions_) {
				transform(location_.begin(), location_.end(), bounds_.begin(), location_.begin(), &clamp);
			}
		}

	private:
		int dimensions_;
	};
}
