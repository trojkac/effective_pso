using System.Runtime.Serialization;
using System.ServiceModel;
using Common;

namespace Service
{
    // NOTE: You can use the "Rename" command on the "Refactor" menu to change the interface name "IService" in both code and config file together.
    [ServiceContract]
    public interface IService
    {
        [OperationContract]
        ParticleState GetKnownBest();

        [OperationContract]
        ParticleState Run(PsoSettings settings);
    }

}
