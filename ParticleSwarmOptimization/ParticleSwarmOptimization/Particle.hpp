#pragma once

#include <vector>
#include <functional>

namespace ParticleSwarmOptimization
{
	public class Particle {
	public:
		virtual void init_location() = 0;
		virtual void init_velocity() = 0;
		virtual std::tuple<std::vector<double>, double> update_personal_best(std::function<double(std::vector<double>)> function) = 0;

		virtual std::vector<Particle> filter_neighbors(std::vector<Particle> other_particles) = 0;

		virtual void update_velocity(std::function<double(std::vector<double>)> function) = 0;
		virtual void translate() = 0;

	protected:
		std::vector<double> location;
		std::tuple<std::vector<double>, double> personal_best;
		std::vector<double> velocity;
	};
}
