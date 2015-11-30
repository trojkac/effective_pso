#pragma once
#include "ParticleSwarmOptimization\Particle.hpp"
namespace ParticleSwarmOptimizationWrapper
{

	public  ref class Particle abstract
	{
	public:
		virtual ParticleSwarmOptimization::Particle* nativeParticle() = 0;
	};

}