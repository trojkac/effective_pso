namespace Common
{
    public class FitnessFunction: IFitnessFunction<double[],double[]>
    {
        private readonly FitnessFunctionEvaluation _evaluate;
        public FitnessFunction(FitnessFunctionEvaluation evaluator)
        {
            _evaluate = evaluator;
            BestEvaluation = null;
        }

        public double[] Evaluate(double[] x)
        {
            var newState = ParticleStateFactory.Create(LocationDim,FitnessDim);
            newState.Location = x;
            newState.FitnessValue = _evaluate(x);

            if (BestEvaluation == null || ((ParticleState) BestEvaluation).IsBetter(newState))
            {
                BestEvaluation = newState;
            }
            return newState.FitnessValue;
        }

        public IState<double[], double[]> BestEvaluation { get; private set; }
        public int LocationDim { get; private set; }
        public int FitnessDim { get; private set; }
    }
}