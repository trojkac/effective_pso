#include "ParticleSwarmOptimization.hpp"
#include "FullyInformedPSOParticle.h"

#include <iostream>

using namespace ParticleSwarmOptimization;
using namespace FullyInformedPSOParicle;

#include <cmath>

double function(std::vector<double> values)
{
	return sin((values[0] * values[0]) + (values[1] * values[1]));
}

int main()
{
	std::vector<Particle*> particles;
	for (int i = 0; i < 20; ++i)
	{
		Particle* p = new FullyInformedParticle(2);
		p->init_velocity();
		p->init_location();
		particles.emplace_back(p);
	}

	PSOAlgorithm algorithm(function, 100);
	auto result = algorithm.run(particles);

	std::cout << "Result: " << std::get<1>(result) << std::endl;
}