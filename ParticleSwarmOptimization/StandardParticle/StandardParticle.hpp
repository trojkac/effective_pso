// ParticleSwarmOptimization.h

#pragma once

#include <algorithm>
#include <random>
#include "../PsoAlgorithm/Particle.hpp"

using namespace ParticleSwarmOptimization;

namespace ParticleSwarmOptimization {

	class StandardParticle : public Particle {
	public:
		explicit StandardParticle(int dimensions);

		void init_location() override;

		void init_velocity() override;

		std::tuple<std::vector<double>, double> update_personal_best(std::function<double(std::vector<double>)> function) override;

		void update_neighborhood(std::vector<Particle*> all_particles) override;

		void update_velocity() override;

		void translate() override;

	private:
		int dimensions_;
	};
}
