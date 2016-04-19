namespace Common
{
    /// <summary>
    /// Objects of classes implementing this interface can be used to calculate fitness value
    /// </summary>
    /// <typeparam name="TLocation">Specifies elements of search space</typeparam>
    /// <typeparam name="TFitness">Specifies elements of fitness value space</typeparam>
    public interface IFitnessFunction<TLocation, TFitness>
    {
        TFitness Evaluate(TLocation x);
        IState<TLocation, TFitness> BestEvaluation { get; }

        int LocationDim { get; }
        int FitnessDim { get; }
    }
}