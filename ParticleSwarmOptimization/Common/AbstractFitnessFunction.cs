using System.Runtime.Serialization;
using Common.Parameters;

namespace Common
{
    [DataContract]
    public abstract class AbstractFitnessFunction : IFitnessFunction<double[],double[]>
    {
        private IOptimization<double[]> _optimization; 
        protected AbstractFitnessFunction(FunctionParameters functionParams)
        {
            Dimension = functionParams.Dimension;
            Coefficients = new double[Dimension];
            functionParams.Coefficients.CopyTo(Coefficients, 0);
            _optimization = PsoServiceLocator.Instance.GetService<IOptimization<double[]>>();
            EvaluationsCount = 0;
        }

        public abstract double[] Calculate(double[] vector);
        [DataMember]
        public int Dimension;
        [DataMember]
        public double[] Coefficients;

        public int EvaluationsCount { get; private set; }

        public double[] Evaluate(double[] x)
        {
            var state = new ParticleState(x,Calculate(x));
            if (BestEvaluation == null || _optimization.IsBetter(state.FitnessValue,BestEvaluation.FitnessValue) < 0)
            {
                BestEvaluation = state;
            }
            EvaluationsCount++;
            return state.FitnessValue;
        }

        public IState<double[], double[]> BestEvaluation { get; private set; }

        public int LocationDim
        {
            get { return Dimension; }
        }

        public int FitnessDim
        {
            get { return 1; }
        }
    }
}