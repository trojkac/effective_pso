#pragma once

#include "../Stdafx.h"

namespace ParticleSwarmOptimization
{
	class Function{
	public:
		Function()
		{
			name_ = "unnamed";
		}
		Function(std::string name)
		{
			name_ = name;
		}
		std::string name(){ return name_; };
		int id(){ return id_; }
		virtual double evaluate(std::vector<double> X) = 0;
		virtual std::tuple<std::vector<double>,double> best_evaluation() = 0;
	protected:
		std::string name_;
	private:
		int id_;
		static int counter_;
	};
}
