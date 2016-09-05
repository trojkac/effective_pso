namespace Common.Parameters
{
    public interface IParametersSerializer<out T>
        where T: class, IParameters
    {
        T Deserialize(string fileName);
    }
}