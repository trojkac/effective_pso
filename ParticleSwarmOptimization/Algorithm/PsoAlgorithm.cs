using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Schema;
using Common;

namespace Algorithm
{
    
	public class PsoAlgorithm
	{   
	    private int _iteration;
	    private readonly IFitnessFunction<double[],double[]> _fitnessFunction;
	    private readonly IParticle[] _particles;
	    private PsoSettings _settings;
        /// <summary>
        /// Creates PsoAlgorithm with specific settings to solve given problem using precreated particles
        /// </summary>
        /// <param name="settings">takes PsoSettings to check what stop conditions are defined.
        ///  PsoAlgorithm looks for TargetValue, Epsilon, TargetValueCondition, IterationsLimit, IterationsLimitCondition
        /// </param>
        /// <param name="fitnessFunction">function whose optimum is to be found</param>
        /// <param name="particles"> particles traversing the search space </param>
        public PsoAlgorithm(PsoSettings settings, IFitnessFunction<double[], double[]> fitnessFunction, IParticle[] particles)
	    {
	        _settings = settings;
	        _fitnessFunction = fitnessFunction;
	        _particles = particles;
	        _iteration = 0;
	    }

	    public IState<double[],double[]> Run()
	    {
	        foreach (var particle in _particles)
	        {
	            particle.UpdatePersonalBest(_fitnessFunction);
	        }
			while (_conditionCheck())
			{
			    foreach (var particle in _particles)
			    {
                    //TODO: convert to one method call
			        particle.Translate();
                    particle.UpdatePersonalBest(_fitnessFunction);
			    }
			    foreach (var particle in _particles)
			    {
			        particle.UpdateNeighborhood(_particles);
                    particle.UpdateVelocity();
			    }
			}
			return _fitnessFunction.BestEvaluation;
		}

	    private bool _conditionCheck()
		{
			return 
                (!_settings.IterationsLimitCondition || _iteration++ < _settings.Iterations)
                && 
                (!_settings.TargetValueCondition ||
                !_fitnessFunction.BestEvaluation.IsCloseToValue(new []{_settings.TargetValue},_settings.Epsilon));
		}

	};
}
