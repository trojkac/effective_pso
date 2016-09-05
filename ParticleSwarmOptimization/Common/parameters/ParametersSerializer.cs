using System.IO;
using System.Xml.Serialization;

namespace Common.Parameters
{
    public class ParametersSerializer<T> : IParametersSerializer<T>
        where T: class, IParameters
    {
        
        public T Deserialize(string fileName)
        {
            var serializer = new XmlSerializer(typeof(T));
            using (var nodeFileReader = new StreamReader(fileName))
            {
                return (T)serializer.Deserialize(nodeFileReader);
            }
        }
    }
}