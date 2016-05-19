﻿using System;
using Common;

namespace Algorithm
{
    public interface IParticle
    {
        int Id { get; }
        ParticleState CurrentState { get; }
        ParticleState PersonalBest { get; }
        void UpdateVelocity();
        void Translate();
        void UpdatePersonalBest(IFitnessFunction<double[], double[]> function);
        void UpdateNeighborhood(IParticle[] allParticles);
        void Init(ParticleState state, double[] velocity, Tuple<double, double>[] bounds = null);

    }
}