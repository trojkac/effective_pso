#pragma once
#include ".\Particle.hpp"
#include "StandardParticle\StandardParticle.hpp"
namespace ParticleSwarmOptimizationWrapper
{
	public ref class FullyInformedParticle : public Particle
	{
		ParticleSwarmOptimization::StandardParticle* _nativeParticle;

	public:
		FullyInformedParticle(int n):
			_nativeParticle(new ParticleSwarmOptimization::StandardParticle(n))
		{
			_nativeParticle->init_location();
			_nativeParticle->init_velocity();
		};
		ParticleSwarmOptimization::Particle* nativeParticle() override
		{
			return _nativeParticle;
		};

	};

}
