using System;
using System.Net;
using System.ServiceModel.Description;
using Common;

namespace Node
{
    public class ProxyParticle
    {
        private int _sourceNodeId;
        private IPsoService _psoClient;
        private ParticleState _bestKnownState;

        public ProxyParticle(NodeInfo sourceNode)
        {
            _sourceNodeId = sourceNode.Id;
            _bestKnownState = new ParticleState(new[]{0.0},double.PositiveInfinity);
            _psoClient = new PsoServiceClient("particleProxtClientTcp", sourceNode.Address.ToString());
        }
        /// <summary>
        /// Function called by the other particle in local swarm to know this particle's personal best
        /// which is personal best of the linked particle in the other swarm
        /// </summary>
        /// <returns></returns>
        public ParticleState GetPersonalBest()
        {
            var s = _psoClient.GetBestState(_sourceNodeId);
            if (s.FitnessValue < _bestKnownState.FitnessValue)
            {
                _bestKnownState = s;
            }
            return _bestKnownState;
        }
        /// <summary>
        /// Function called from remote node via IPsoManager
        /// </summary>
        /// <returns>Best position of the local connected particle</returns>
        public ParticleState GetLocalBest()
        {
            //TODO: It has to get best position of the linked particle somehow...
            throw new NotImplementedException();
            
        } 



    }
}