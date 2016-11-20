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
        private object _randomLock;
        private static object _generatorLock = new object();
        private Random Random;


        private RandomGenerator(int? seed = null)
        {
            _seed = seed ?? DateTime.Now.Millisecond;
            Random = new Random(_seed.Value);
            _randomLock = new object();
        }

        public double[] RandomVector(int dim, double min = -2, double max = 2)
        {
            return RandomVector(dim, Enumerable.Repeat(new DimensionBound(min, max), dim).ToArray());
        }

        public double[] RandomVector(int dim, DimensionBound[] bounds)
        {
            lock (_randomLock)
            {
                if (bounds == null) throw new ArgumentNullException();
                var v = new double[dim];
                for (var i = 0; i < dim; i++)
                {
                    v[i] = Random.NextDouble() * (bounds[i].Max - bounds[i].Min) + bounds[i].Min;
                }
                return v;
            }
        }

        public int RandomInt(int min, int max)
        {
            lock(_randomLock)
            {
                return Random.Next(min,max);
            }
        }

        public static RandomGenerator GetInstance(int? seed = null)
        {
            lock (_generatorLock)
            {
                if (_generator != null && seed != _seed && seed != null)
                {
                    throw new ArgumentException("Generator already initialized with other seed value");
                }
                return _generator ?? (_generator = new RandomGenerator(seed));
            }
        }
    }
}
