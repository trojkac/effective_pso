#pragma once
#include <functional>

using namespace Common;

std::tuple<std::vector<double>, double> particle_state_to_tuple(ParticleState^ src);
ParticleState^ tuple_to_particle_state(std::tuple<std::vector<double>, double> src);