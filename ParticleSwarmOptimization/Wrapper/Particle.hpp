#pragma once
#include "native\Particle.hpp"

namespace ParticleSwarmOptimizationWrapper
{
	public ref class ParticleWrapper abstract
	{
	public:
		virtual ParticleSwarmOptimization::Particle* nativeParticle() = 0;

		void bounds(System::Collections::Generic::List<System::Tuple<double, double>^>^ bounds)
		{
			std::vector<std::tuple<double, double>> nativeBounds;
			for each (System::Tuple<double,double>^ var in bounds)
			{
				nativeBounds.emplace_back(std::make_tuple(var->Item1, var->Item2));
			}
			nativeParticle()->bounds(nativeBounds);
		}
	};

}
