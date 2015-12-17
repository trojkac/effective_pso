#include "stdafx.h"
#include ".\StandardParticle.hpp"


using namespace System;
using namespace System::Collections::Generic;
using namespace System::Runtime::InteropServices;
using namespace Common;

std::tuple<std::vector<double>, double> particle_state_to_tuple(ParticleState^ src)
{
	array<double>^ a = src->Location;
	auto v = std::vector<double>(a->Length);
	{
		pin_ptr<double> x(&a[0]);
		std::copy(
			static_cast<double*>(x),
			static_cast<double*>(x + a->Length),
			v.begin()
			);
	}
	return std::make_tuple(v, src->FitnessValue);

}
ParticleState^ tuple_to_particle_state(std::tuple<std::vector<double>, double> src)
{
	array<double> ^vals = gcnew array<double>(std::get<0>(src).size());
	Marshal::Copy(IntPtr((void*)std::get<0>(src).data()), vals, 0, std::get<0>(src).size());
	return gcnew ParticleState(vals, std::get<1>(src));
}