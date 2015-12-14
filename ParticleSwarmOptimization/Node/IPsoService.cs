
using System.ServiceModel;
using Common;

namespace Node
{
    [ServiceContract]
    public interface IPsoService
    {
        /// <summary>
        /// Returns best known position of particle which is in neighborhood (in terms of PSO algorithm) o
        /// </summary>
        /// <param name="nodeId">
        ///     Id of node which is sender of the request
        /// </param>
        /// <returns>
        ///     Best known position of the particle 
        /// </returns>
        [OperationContract]
        ParticleState GetBestState(int nodeId);
    }
}