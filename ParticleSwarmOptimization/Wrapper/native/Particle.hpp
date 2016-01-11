#pragma once

#include "../Stdafx.h"
#include "Function.hpp" 
namespace ParticleSwarmOptimization
{
	class Particle {
	public:
		virtual ~Particle()
		{
		}
		Particle()
		{
			id_ = ++counter_;
		}
		virtual void init_location() = 0;
		virtual void init_velocity() = 0;
		virtual std::tuple<std::vector<double>, double> update_personal_best(Function *function) = 0;

		virtual std::tuple<std::vector<double>, double> get_personal_best()
		{
			return personal_best_;
		}
		virtual void update_neighborhood(std::vector<Particle*> all_particles) = 0;
		virtual void update_velocity() = 0;
		virtual void translate() = 0;
		void bounds(std::vector<std::tuple<double, double>> bounds);
		int id() const;
	protected:
		std::vector<double> location_;
		std::tuple<std::vector<double>, double> personal_best_;
		std::vector<double> velocity_;
		std::vector<std::tuple<double, double>> bounds_;
		std::vector<Particle*> neighborhood_;

		static double clamp(double, std::tuple<double,double>);
	private:
		int id_;
		static int counter_;
	};
	inline void Particle::bounds(std::vector<std::tuple<double, double>> bounds)
	{
		bounds_ = bounds;
	}

	inline  double Particle::clamp(double x, std::tuple<double, double> bounds)
	{
		return std::max(std::min(x, std::get<1>(bounds)), std::get<0>(bounds));
	}

	inline int Particle::id() const
	{
		return id_;
	}

}
