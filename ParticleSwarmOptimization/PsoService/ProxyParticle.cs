
using System;
using System.Diagnostics;
using System.Linq;
using System.ServiceModel;
using System.ServiceModel.Description;
using Algorithm;
using Common;

namespace PsoService
{
    
    public delegate void ParticleCommunicationBreakdown();
    public class ProxyParticle : Particle
    {

        private ProxyManager _proxyManager;

        public NetworkNodeInfo RemoteNode { get { return _proxyManager.RemoteNode; } }
        public Uri Address { get { return _proxyManager.Address; } }
        public Uri RemoteAddress { get { return _proxyManager.RemoteAddress; } }

        public ProxyParticle(ProxyManager proxyManager)
        {
            _proxyManager = proxyManager;
            proxyManager.RestartState();
        }

        public void Open()
        {
            _proxyManager.Open();
        }

        public void Close()
        {
            _proxyManager.Close();
        }


        public override void UpdateNeighborhood(IParticle[] allParticles)
        {
            if (_coupledParticle == null)
            {
                _coupledParticle = allParticles.Where(p => p.Id != Id).First();
                if (_coupledParticle != null)
                {
                    _proxyManager.UpdateBestState(_coupledParticle.PersonalBest);
                }
            }
        }

        public override void Init(ParticleState state, double[] velocity, DimensionBound[] bounds = null)
        {

        }

        public override int Id
        {
            get { return _id; }
        }

        public override void UpdateVelocity(IState<double[], double[]> globalBest)
        {
            _proxyManager.UpdateBestState((ParticleState)globalBest);
        }

        public override void Transpose(IFitnessFunction<double[], double[]> function)
        {
        }


        private int _getBestCounter = 0;
        private const int RemoteCheckInterval = 200;
        private IParticle _coupledParticle;
        public override ParticleState PersonalBest
        {
            get
            {
                if (_getBestCounter == RemoteCheckInterval)
                {
                    _getBestCounter = 0;
                    _proxyManager.GetRemoteBestState();
                }
                _getBestCounter++;
                return _proxyManager.GetBestState() ;

            }
        }
    }
}