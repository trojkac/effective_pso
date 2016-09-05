﻿using System;
using System.Collections.Generic;
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
	    private readonly IFitnessFunction<double[],double[]> _fitnessFunction;
	    private readonly IParticle[] _particles;
	    private ILogger _logger;
	    private PsoParameters _parameters;
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
	    }

	    public IState<double[],double[]> Run()
	    {
	        foreach (var particle in _particles)
	        {
	            particle.Transpose(_fitnessFunction);
                particle.UpdateNeighborhood(_particles);

	        }
			while (_conditionCheck())
			{
			    foreach (var particle in _particles)
			    {
                    particle.Transpose(_fitnessFunction);
			    }
			    foreach (var particle in _particles)
			    {
                    particle.UpdateVelocity();
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
			return 
                (!_parameters.IterationsLimitCondition || _iteration++ < _parameters.Iterations)
                && 
                (!_parameters.TargetValueCondition ||
                !(PsoServiceLocator.Instance.GetService<IOptimization<double[]>>().AreClose(new []{_parameters.TargetValue},_fitnessFunction.BestEvaluation.FitnessValue,_parameters.Epsilon)));
		}

	};
}
