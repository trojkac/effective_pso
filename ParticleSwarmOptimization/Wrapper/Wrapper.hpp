// Wrapper.hpp

#pragma once

#include <functional>
#include <vector>
#include "native\PsoAlgorithm.hpp"
#include "native\Particle.hpp"
#include "Particle.hpp"
#include "WrapperHelper.hpp"



using namespace System;
using namespace System::Collections::Generic;
using namespace System::Runtime::InteropServices;
using namespace Common;
namespace ParticleSwarmOptimizationWrapper {

	typedef double(*UnmanagedFitnessfunction)(std::vector<double>); // typedef of unmanaged fitness function

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
		static PSOAlgorithm^ GetAlgorithm(int iterations, double targetValue, double epsilon, FitnessFunction^ fitnessFunction)
		{
			auto algorithm = gcnew PSOAlgorithm(fitnessFunction);
			algorithm->managedFitness = gcnew ManagedFitnessFunction(algorithm, &nativeFunction);
			auto funcPtr = Marshal::GetFunctionPointerForDelegate(algorithm->managedFitness);
			auto unmanagedFitness = static_cast<UnmanagedFitnessfunction>(funcPtr.ToPointer());
			algorithm->_algorithm = new ParticleSwarmOptimization::PSOAlgorithm(unmanagedFitness,iterations,targetValue,epsilon);
			return algorithm;
		}
		static PSOAlgorithm^ GetAlgorithm(double targetValue, double epsilon,FitnessFunction^ fitnessFunction)
		{
			auto algorithm = gcnew PSOAlgorithm(fitnessFunction);
			algorithm->managedFitness = gcnew ManagedFitnessFunction(algorithm, &nativeFunction);
			auto funcPtr = Marshal::GetFunctionPointerForDelegate(algorithm->managedFitness);
			auto unmanagedFitness = static_cast<UnmanagedFitnessfunction>(funcPtr.ToPointer());
			algorithm->_algorithm = new ParticleSwarmOptimization::PSOAlgorithm(unmanagedFitness, targetValue, epsilon);
			return algorithm;
		}
		ParticleState^ Run(List<Particle^>^ particles)
		{
			std::vector<ParticleSwarmOptimization::Particle*> stdParticles;
			for each (auto particle in particles)
			{
				ParticleSwarmOptimization::Particle* native = particle->nativeParticle();
				stdParticles.emplace_back(native);
			}
			auto result = _algorithm->run(stdParticles);
			return tuple_to_particle_state(result);
		}
	};
}
