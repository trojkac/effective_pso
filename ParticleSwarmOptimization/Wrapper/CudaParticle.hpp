#pragma once

#include "native\Particle.hpp"
#include "native\Function.hpp"

#include <tuple>
#include <vector>

namespace ParticleSwarmOptimization
{
    class CudaParticle : public ParticleSwarmOptimization::Particle
    {
    private:
        std::tuple<std::vector<double>, double>* local_endpoint_;

        std::tuple<std::vector<double>, double>* remote_endpoint_;

        ParticleSwarmOptimization::Particle* coupled_particle_;

    public:
        CudaParticle(std::tuple<std::vector<double>, double>* remote_endpoint,
                     std::tuple<std::vector<double>, double>* local_endpoint) :
                     remote_endpoint_(remote_endpoint), local_endpoint_(local_endpoint)
        {}

        void update_neighborhood(std::vector<ParticleSwarmOptimization::Particle*> all_particles) override
        {
            int i = 0;

            while(coupled_particle_ == NULL || coupled_particle_->id() == id())
            {
                coupled_particle_ = all_particles[i++];
            }
        }

        std::tuple<std::vector<double>, double> get_personal_best() override
        {
            return personal_best_;
        }

        std::tuple<std::vector<double>, double> update_personal_best(ParticleSwarmOptimization::Function *function) override
        {
            auto remote_best = *remote_endpoint_;

            if(std::get<1>(personal_best_) > std::get<1>(remote_best))
            {
                personal_best_ = remote_best;
                local_endpoint_ = new std::tuple<std::vector<double>, double>(personal_best_);
            }

            if(coupled_particle_ != NULL)
            {
                auto coupled_best = coupled_particle_->get_personal_best();
                if(std::get<1>(personal_best_) > std::get<1>(coupled_best))
                {
                    personal_best_ = coupled_best;
                    local_endpoint_ = new std::tuple<std::vector<double>, double>(personal_best_);
                }
            }

            return personal_best_;
        }

        void update_velocity() {}

        void translate() {}

        void init_location() {}

        void init_velocity() {}
    };
};