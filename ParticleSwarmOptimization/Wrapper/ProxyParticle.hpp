#include <msclr\auto_gcroot.h>
#include <functional>
#include "Particle.hpp"
#include "PsoAlgorithm\Particle.hpp"
#include "WrapperHelper.hpp"

//native
namespace ParticleSwarmOptimization
{

	class ProxyParticleBox
	{
	public:
		msclr::auto_gcroot<PsoService::ProxyParticleService^> proxyService;

		ProxyParticleBox(PsoService::ProxyParticleService^ proxy) :
			proxyService(proxy)
		{
		}

		std::tuple<std::vector<double>, double> get_remote_best()
		{
			auto remote_best = proxyService->GetBestState();

		}
	};

	class ProxyParticle : public  ParticleSwarmOptimization::Particle
	{

	public:

		ProxyParticle(int dimensions, ProxyParticleBox* box) :
			dimensions_(dimensions),
			proxy_particle_(box)
		{
			personal_best_ = std::make_tuple(std::vector<double>(2), -std::numeric_limits<double>::infinity());
			box->proxyService->Open();
		}

		void update_neighborhood(std::vector<Particle*> all_particles)
		{
			if (coupled_particle_ == NULL)
			{
				//TODO: take random particle
				coupled_particle_ = all_particles[0];
			}
		}

		std::tuple<std::vector<double>, double> get_personal_best() override
		{
			return personal_best_;
		}
		std::tuple<std::vector<double>, double> update_personal_best(std::function<double(std::vector<double>)> function)
		{
			auto remote_best = particle_state_to_tuple(proxy_particle_->proxyService->GetRemoteBest());
			if (std::get<1>(personal_best_) < std::get<1>(remote_best))
			{
				personal_best_ = remote_best;
				proxy_particle_->proxyService->UpdateBestState(tuple_to_particle_state(personal_best_));
			}
			auto coupled_best = coupled_particle_->get_personal_best();
			if (std::get<1>(personal_best_) < std::get<1>(coupled_best))
			{
				personal_best_ = coupled_best;
				proxy_particle_->proxyService->UpdateBestState(tuple_to_particle_state(personal_best_));
			}
			return personal_best_;
		}

		void update_velocity()
		{
		}
		void translate()
		{
		}
		void init_location()
		{
		}
		void init_velocity()
		{
		}
		~ProxyParticle()
		{
			proxy_particle_->proxyService->Close();
		}
	private:
		ProxyParticleBox* proxy_particle_;
		int dimensions_;

		ParticleSwarmOptimization::Particle* coupled_particle_;
	};
}