using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;

namespace Common
{
    [DataContract]
    public class ParticleState : IState<double[],double[]>,ICloneable
    {
        [DataMember]
        public double[] FitnessValue { get; set; }
        [DataMember]
        public double[] Location { get; set; }

        private IMetric<double[]> _metric;
        private IOptimization<double[]> _optimization; 
 
        public ParticleState(IOptimization<double[]> optimization1, IMetric<double[]> metric)
        {
            _optimization = optimization1;
            _metric = metric;
        }

        public bool IsBetter(ParticleState otherState)
        {
            return _optimization.IsBetter(FitnessValue, otherState.FitnessValue) < 0;
        }

        public bool IsCloseToValue(double[] value, double epsilon)
        {
            // TODO: Check for multiple dimensions
            return Math.Abs(FitnessValue[0] - value[0]) < epsilon;
        }

        public double Distance(ParticleState otherState)
        {
            return _metric.Distance(Location, otherState.Location);
        }

        public double[] VectorTo(ParticleState otherState)
        {
            return _metric.VectorBetween(Location, otherState.Location);
        }

        public object Clone()
        {
            return new ParticleState(_optimization,_metric){Location = (double[])Location.Clone(),FitnessValue = (double[])FitnessValue.Clone()};
        }
    }

//    public class GenericParticleState<TLocation, TFitness> : ParticleState, IState<TLocation, TFitness>
//    {
//        
//        private IMetric<TLocation> _metric;
//        private IOptimization<TFitness> _optimization;
//
//        public GenericParticleState(IMetric<TLocation> metric, IOptimization<TFitness> optimization)
//        {
//            _metric = metric;
//            _optimization = optimization;
//        }
//
//        public override bool IsBetter(ParticleState otherState)
//        {
//            throw new NotImplementedException();
//        }
//
//        public override bool IsClose(ParticleState otherState, double epsilon)
//        {
//            throw new NotImplementedException();
//        }
//
//        public override double Distance(ParticleState otherState)
//        {
//            
//        }
//
//        public TLocation Location { get; set; }
//        public TFitness FitnessValue { get; set; }
//    }
}