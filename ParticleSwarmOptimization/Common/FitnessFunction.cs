namespace Common
{
    public class FitnessFunction: IFitnessFunction<double[],double[]>
    {
        private readonly FitnessFunctionEvaluation _evaluate;
        private readonly IOptimization<double[]> _optimization;
        public FitnessFunction(FitnessFunctionEvaluation evaluator, IOptimization<double[]> optimization = null)
        {
            _evaluate = evaluator;
            _optimization = PsoServiceLocator.Instance.GetService<IOptimization<double[]>>();
            BestEvaluation = null;
        }

        public double[] Evaluate(double[] x)
        {
            var newState = new ParticleState(x, _evaluate(x));
            if (BestEvaluation == null ||  _optimization.IsBetter(newState.FitnessValue,BestEvaluation.FitnessValue) < 0)
            {
                BestEvaluation = newState;
            }
            return newState.FitnessValue;
        }

        public IState<double[], double[]> BestEvaluation { get; private set; }
        public int LocationDim { get; set; }
        public int FitnessDim { get; set; }
    }
}