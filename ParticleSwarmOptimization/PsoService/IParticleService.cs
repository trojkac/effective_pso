using System.ServiceModel;
using Common;

namespace PsoService
{
    [ServiceContract]
    public interface IParticleService
    {
        /// <summary>
        ///     Returns best known position of particle which is in neighborhood (in terms of PSO algorithm) o
        /// </summary>
        /// <returns>
        ///     Best known position of the particle 
        /// </returns>
        [OperationContract]
        ParticleState GetBestState();

        void UpdateBestState(ParticleState state);

        void RestartState();
    }
}