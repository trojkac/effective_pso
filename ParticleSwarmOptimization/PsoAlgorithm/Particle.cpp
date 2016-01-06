#include "Particle.hpp"
#include <algorithm>

int ParticleSwarmOptimization::Particle::counter_ = 0;

inline ParticleSwarmOptimization::Particle::Particle()
{
	id_ = ++counter_;
}

inline std::tuple<std::vector<double>, double> ParticleSwarmOptimization::Particle::get_personal_best()
{
	return personal_best_;
}

inline void ParticleSwarmOptimization::Particle::bounds(std::vector<std::tuple<double, double>> bounds)
{
	bounds_ = bounds;
}

inline  double ParticleSwarmOptimization::Particle::clamp(double x, std::tuple<double, double> bounds)
{
	return std::max(std::min(x, std::get<1>(bounds)), std::get<0>(bounds));
}

inline int ParticleSwarmOptimization::Particle::id() const
{
	return id_;
}