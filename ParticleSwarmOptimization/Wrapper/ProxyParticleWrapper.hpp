#pragma once
#include "PsoAlgorithm\Particle.hpp"
#include <msclr\auto_gcroot.h>
#include "Particle.hpp"
#include ".\ProxyParticle.hpp"

using namespace System;
using namespace System::Collections::Generic;
using namespace System::Runtime::InteropServices;
using namespace Common;

namespace ParticleSwarmOptimizationWrapper
{
	public ref class ProxyParticle : public Particle
	{
		ParticleSwarmOptimization::ProxyParticle* _nativeParticle;

	public:
		ProxyParticle(int n, PsoService::ProxyParticle^ service)
		{
			ParticleSwarmOptimization::ProxyParticleBox* box = new ParticleSwarmOptimization::ProxyParticleBox(service);
			_nativeParticle = new ParticleSwarmOptimization::ProxyParticle(n, box);
			_nativeParticle->init_location();
			_nativeParticle->init_velocity();
		};

		ParticleSwarmOptimization::Particle* nativeParticle() override
		{
			return _nativeParticle;
		};
	};
}