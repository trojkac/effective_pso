#pragma once

#include "Particle.hpp"
#include "CudaParticle.hpp"

namespace ParticleSwarmOptimizationWrapper
{
    public ref class CudaParticleWrapper : public ParticleWrapper
    {
        ParticleSwarmOptimization::CudaParticle* _nativeParticle;

    public:
        CudaParticleWrapper(std::tuple<std::vector<double>, double>* remoteEndpoint,
                            std::tuple<std::vector<double>, double>* localEndpoint) :
                            _nativeParticle(new ParticleSwarmOptimization::CudaParticle(remoteEndpoint, localEndpoint))
        {};

        ParticleSwarmOptimization::Particle* nativeParticle() override
        {
            return _nativeParticle;
        };
    };

    public ref class CudaPraticleWrapperFactory
    {
    public:
        static CudaParticleWrapper^ Create(void* remoteEndpoint,
                                           void* localEndpoint)
        {
            return gcnew CudaParticleWrapper((std::tuple<std::vector<double>, double>*)remoteEndpoint, 
                                             (std::tuple<std::vector<double>, double>*)localEndpoint);
        }
    };
}