using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common
{
    public class RandomGenerator
    {
        private static RandomGenerator _generator;
        private static int? _seed;
        private Random Random;


        private RandomGenerator(int? seed = null)
        {
            _seed = seed ?? DateTime.Now.Millisecond;
            Random = new Random(_seed.Value);
        }

        public double[] RandomVector(int dim, double min=-2, double max=2)
        {
            var v = new double[dim];
            for (var i = 0; i < dim; i++)
            {
                v[i] = Random.NextDouble()*(max - min) + min;
            }
            return v;
        }

        public double[] RandomVector(int dim, DimensionBound[] bounds)
        {
            if(bounds == null) throw new ArgumentNullException();
            var v = new double[dim];
            for (var i = 0; i < dim; i++)
            {
                v[i] = Random.NextDouble() * (bounds[i].Max - bounds[i].Min) + bounds[i].Min;
            }
            return v;
        }
        public static  RandomGenerator GetInstance(int? seed = null)
        {
            if (_generator != null && seed != _seed && seed != null)
            {
                throw new ArgumentException("Generator already initialized with other seed value");
            }
            return _generator ?? (_generator = new RandomGenerator(seed));
        }
    }
}
