using System.Collections.Generic;
using System.Dynamic;
using System.Text;
using System.Threading.Tasks;

namespace Common
{
    public sealed class ParticleStateFactory
    {
        private static IMetric<double[]> _metric = new Euclidean();
        private static IOptimization<double[]> _optimization = new FirstValueOptimization(); 
        public static ParticleState Create(int locationDim, int fitnessDim)
        {
            var rg = RandomGenerator.GetInstance();
            return new ParticleState(_optimization, _metric)
            {
                FitnessValue = rg.RandomVector(fitnessDim),
                Location = rg.RandomVector(locationDim)
            };
        }
    }
}
