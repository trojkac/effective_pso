﻿using System;
using Common;

namespace Algorithm
{
    public interface IParticle
    {
        int Id { get; }
        ParticleState CurrentState { get; }
        ParticleState PersonalBest { get; }
        void UpdateVelocity(IState<double[], double[]> globalBest);
        void Transpose(IFitnessFunction<double[], double[]> function);
        void UpdateNeighborhood(IParticle[] allParticles);
        void Init(ParticleState state, double[] velocity, DimensionBound[] bounds = null);
        void InitializeVelocity(IParticle particle);
    }
}