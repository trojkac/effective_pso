namespace Common
{
    /// <summary>
    /// Interface used to store information about particle state
    /// </summary>
    /// <typeparam name="TLocation">type of an elements in search space</typeparam>
    /// <typeparam name="TFitness">type of an elements of fitness value space</typeparam>
    public interface IState<TLocation,TFitness>
    {
        TLocation Location { get; set; }
        TFitness FitnessValue { get; set; }
    }
}