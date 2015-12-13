#include "ParticleSwarmOptimization.hpp"
#include "StandardParticle.hpp"

#include <iostream>

using namespace ParticleSwarmOptimization;

double function(std::vector<double> values)
{
	return -1*(values[0]*values[0] - 9)*(values[0] - 5)*(values[1] - 2)*(values[1]* values[1]* values[1] + 25);
}

int main()
{
	std::vector<Particle*> particles;
	std::vector<std::tuple<double, double>> bounds;
	bounds.emplace_back(std::tuple<double, double>(-4, 4));
	bounds.emplace_back(std::tuple<double, double>(-4, 4));

	for (int i = 0; i < 20; ++i)
	{
		Particle* p = new StandardParticle(2);
		p->bounds(bounds);
		p->init_velocity();
		p->init_location();
		particles.emplace_back(p);
	}

	PSOAlgorithm algorithm(function, 1000);
	auto result = algorithm.run(particles);

	std::cout << "Result: " << std::get<1>(result) << std::endl;
	std::cout << "At: (" << std::get<0>(result)[0] << ", " << std::get<0>(result)[1]<<")"<< std::endl;

}