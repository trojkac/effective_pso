using System.Net;
using Common;

namespace Distributor
{
    public interface IDistributor
    {
        double RunDispersed(FitnessFunction function, PsoSettings settings, params IPAddress[] servicesAddresses);
    }
}