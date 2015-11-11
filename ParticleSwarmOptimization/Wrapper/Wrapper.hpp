// Wrapper.hpp

#pragma once

#include <functional>
#include "ParticleSwarmOptimization\Particle.hpp"
#include "ParticleSwarmOptimization\ParticleSwarmOptimization.hpp"
#include ".\Particle.hpp"

using namespace System;
using namespace System::Collections::Generic;
using namespace System::Runtime::InteropServices;

namespace ParticleSwarmOptimizationWrapper {

	typedef double(*UnmanagedFitnessfunction)(std::vector<double>); // typedef of unmanaged fitness function
	public delegate double FitnessFunction(array<double>^ values); // delegate to use in client applications

	public ref class PSOAlgorithm
	{
		ParticleSwarmOptimization::PSOAlgorithm* _algorithm;
		FitnessFunction^ _fitnessFunction;
		//managed delegate - used to convert 'outside' delegate to unmanaged delegate
		delegate double ManagedFitnessFunction(std::vector<double> args);
		ManagedFitnessFunction^ managedFitness;
		
		double nativeFunction(std::vector<double> values)
		{
			array<double> ^vals = gcnew array<double>(values.size());
			Marshal::Copy(IntPtr((void*)values.data()), vals, 0, values.size());
			return _fitnessFunction(vals);
		}

		PSOAlgorithm(FitnessFunction^ fitnessFunction) :
			_fitnessFunction(fitnessFunction)
		{
		}
	public:

		static PSOAlgorithm^ GetAlgorithm(int iterations, FitnessFunction^ fitnessFunction)
		{
			auto algorithm = gcnew PSOAlgorithm(fitnessFunction);
			algorithm->managedFitness = gcnew ManagedFitnessFunction(algorithm, &nativeFunction);
			auto funcPtr = Marshal::GetFunctionPointerForDelegate(algorithm->managedFitness);
			auto unmanagedFitness = static_cast<UnmanagedFitnessfunction>(funcPtr.ToPointer());
			algorithm->_algorithm = new ParticleSwarmOptimization::PSOAlgorithm(unmanagedFitness, iterations);
			return algorithm;
		}

		Tuple<List<Double>^, Double>^ Run(List<Particle^>^ particles)
		{
			std::vector<ParticleSwarmOptimization::Particle*> stdParticles;
			for each (auto particle in particles)
			{
				ParticleSwarmOptimization::Particle* native = particle->nativeParticle();
				stdParticles.emplace_back(native);
			}
			auto result = _algorithm->run(stdParticles);
			return gcnew Tuple<List<Double>^, Double>(gcnew List<Double>(), std::get<1>(result));
		}
	};


}
