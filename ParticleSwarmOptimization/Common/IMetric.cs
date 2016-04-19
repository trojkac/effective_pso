namespace Common
{
    /// <summary>
    ///     Specifies metrics of the search space
    /// </summary>
    /// <typeparam name="TLocation">Type of search space(e.g. int[], double[])</typeparam>
    public interface IMetric<TLocation>
    {
        double Distance(TLocation from, TLocation to);
        TLocation VectorBetween(TLocation from, TLocation to);
    }
}