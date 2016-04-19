namespace Common
{
    /// <summary>
    ///     Specifies what type of optimization would be performed (minimization/maximization)
    /// </summary>
    /// <typeparam name="TFitness">Type of fitness values to be compared</typeparam>
    public interface IOptimization<in TFitness>
    {
        /// <summary>
        /// Gives answer on which fitness value is better
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>
        /// ** negative value if a is better,
        /// ** zero if a is equal to b, 
        /// ** positive value if b is better
        /// </returns>
        int IsBetter(TFitness a, TFitness b);
    }
}