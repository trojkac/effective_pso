#pragma once
#include ".\Particle.hpp"
#include "FullyInformedPSOParicle\FullyInformedPSOParticle.h"
namespace ParticleSwarmOptimizationWrapper
{
	public ref class FullyInformedParticle : public Particle
	{
		FullyInformedPSOParicle::FullyInformedParticle* _nativeParticle;

	public:
		FullyInformedParticle(int n):
			_nativeParticle(new FullyInformedPSOParicle::FullyInformedParticle(n))
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
