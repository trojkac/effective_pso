using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Schema;
using Common;
using Common.Parameters;

namespace Algorithm
{
    
	public class PsoAlgorithm
	{   
	    private int _iteration;
        private int _iterationsSinceImprovement = 0;
	    private IState<double[],double[]> _globalBest;

	    private readonly IFitnessFunction<double[],double[]> _fitnessFunction;
	    private readonly IParticle[] _particles;
	    private ILogger _logger;
	    private PsoParameters _parameters;

	    private readonly IOptimization<double[]> _optimizer;


	    /// <summary>
	    /// Creates PsoAlgorithm with specific parameters to solve given problem using precreated particles
	    /// </summary>
	    /// <param name="parameters">takes PsoParameters to check what stop conditions are defined.
	    ///  PsoAlgorithm looks for TargetValue, Epsilon, TargetValueCondition, IterationsLimit, IterationsLimitCondition
	    /// </param>
	    /// <param name="fitnessFunction">function whose optimum is to be found</param>
	    /// <param name="particles"> particles traversing the search space </param>
	    /// <param name="logger"></param>
        public PsoAlgorithm(PsoParameters parameters, IFitnessFunction<double[], double[]> fitnessFunction, IParticle[] particles, ILogger logger = null)
	    {
	        _parameters = parameters;
	        _fitnessFunction = fitnessFunction;
	        _particles = particles;
	        _iteration = 0;
            _logger = logger;
	        _optimizer = PsoServiceLocator.Instance.GetService<IOptimization<double[]>>();
	    }

	    public IState<double[],double[]> Run()
	    {

	        for(var i = 0; i < _particles.Length; i++)
	        {
	            var particle = _particles[i];
	            var j = i;
	            while( _parameters.ParticlesCount != 1 && ( j == i || _particles[j].CurrentState.Location == null ) )
	            {
	                j = RandomGenerator.GetInstance().RandomInt(0, _particles.Length);
	            }
	            particle.Transpose(_fitnessFunction);
                particle.UpdateNeighborhood(_particles);

	            particle.InitializeVelocity(_particles[j]);
	        }
	        var _currentBest = GetCurrentBest();
            _globalBest = new ParticleState(_currentBest.Location,_currentBest.FitnessValue);
			while (_conditionCheck())
			{
			    foreach (var particle in _particles)
			    {
                    particle.Transpose(_fitnessFunction);
			    }
			    foreach (var particle in _particles)
			    {
                    particle.UpdateVelocity(_globalBest);
			    }
			    if (_logger != null)
			    {
			        _logger.Log(String.Format("ITERATION {0}:",_iteration));
			        foreach (var particle in _particles)
			        {
			            _logger.Log((Particle)particle);
			        }
			    }
			}
			return _fitnessFunction.BestEvaluation;
		}

	    private bool _conditionCheck()
	    {
	        var currentBest = GetCurrentBest();
            if (_optimizer.IsBetter(currentBest.FitnessValue, _globalBest.FitnessValue) < 0)
            {
                _iterationsSinceImprovement = 0;
                _globalBest = new ParticleState(currentBest.Location, currentBest.FitnessValue);
            }
            else
            {
                _iterationsSinceImprovement++;
            }
			return 
                (!_parameters.IterationsLimitCondition || _iteration++ < _parameters.Iterations)
                && 
                (!_parameters.TargetValueCondition ||
                !(_optimizer.AreClose(new []{_parameters.TargetValue},_fitnessFunction.BestEvaluation.FitnessValue,_parameters.Epsilon)))
                &&
                _iterationsSinceImprovement < _parameters.PsoIterationsToRestart           
                ;
		}

	    private IState<double[], double[]> GetCurrentBest()
	    {
            var currentBest = _particles[0].CurrentState;
            foreach (var particle in _particles)
            {
                if (particle.CurrentState.FitnessValue != null && _optimizer.IsBetter(currentBest.FitnessValue, particle.CurrentState.FitnessValue) > 0)
                {
                    currentBest = new ParticleState(particle.CurrentState.Location, particle.CurrentState.FitnessValue);
                }
            }
            return currentBest;
	    }
    };

    
}
