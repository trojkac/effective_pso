#pragma once

#include <vector>
#include <functional>

namespace ParticleSwarmOptimization
{
	class Particle {
	public:
		virtual ~Particle()
		{
		}

		virtual void init_location() = 0;
		virtual void init_velocity() = 0;
		virtual std::tuple<std::vector<double>, double> update_personal_best(std::function<double(std::vector<double>)> function) = 0;
		
		virtual std::tuple<std::vector<double>, double> get_personal_best()
		{
			return personal_best_;
		}

		virtual void update_velocity(std::vector<Particle*> all_particles) = 0;
		virtual void translate() = 0;

	protected:
		std::vector<double> location_;
		std::tuple<std::vector<double>, double> personal_best_;
		std::vector<double> velocity_;
	};
}
