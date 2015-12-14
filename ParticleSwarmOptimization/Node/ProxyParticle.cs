using System;
using System.Net;
using Common;

namespace Node
{
    public class ProxyParticle
    {
        private int _sourceNodeId;
        private EndPoint _targetNodeAddress;
        private ParticleState BestKnownState;
        public ProxyParticle(int sourceNodeId, EndPoint targetNodeAddress)
        {
            _sourceNodeId = sourceNodeId;
            _targetNodeAddress = targetNodeAddress;
        }
        /// <summary>
        /// Function called by the other particle in local swarm to know this particle's personal best
        /// which is personal best of the linked particle in the other swarm
        /// </summary>
        /// <returns></returns>
        public ParticleState GetPersonalBest()
        {
            //TODO: Call to _targetNodeAddress;
            //TODO: It has to be wrapped by C++ particle inheriting from Particle abstract class
            throw new NotImplementedException();
        }
        /// <summary>
        /// Function called from remote node
        /// </summary>
        /// <returns>Best position of the local connected particle</returns>
        public ParticleState GetLocalBest()
        {
            //TODO: It has to get best position of the linked particle somehow...
            throw new NotImplementedException();
            
        } 



    }
}