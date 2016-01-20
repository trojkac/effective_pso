#pragma once

#include "CudaParticle.hpp"

namespace CudaPsoWrapper
{
    public ref class CudaParticle : public ParticleSwarmOptimizationWrapper::ParticleWrapper
    {
        CudaPSO::CudaParticle* _nativeParticle;

    public:
        CudaParticle(std::tuple<std::vector<double>, double>* remoteEndpoint,
                     std::tuple<std::vector<double>, double>* localEndpoint) :
                     _nativeParticle(new CudaPSO::CudaParticle(remoteEndpoint, localEndpoint))
        {};

        ParticleSwarmOptimization::Particle* nativeParticle() override
        {
            return _nativeParticle;
        };
    };
}