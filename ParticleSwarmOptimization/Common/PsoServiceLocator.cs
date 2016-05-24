using System;
using System.Collections.Generic;

namespace Common
{
    public class PsoServiceLocator
    {
        private readonly Dictionary<Type, object> _servicesDictionary;
        private static readonly PsoServiceLocator _instance = new PsoServiceLocator();
        public static PsoServiceLocator Instance
        {
            get
            {
                return _instance;
            }
        }
        private PsoServiceLocator()
        {
            _servicesDictionary = new Dictionary<Type, object>();
            Register<IOptimization<double[]>>(new FirstValueOptimization());
            Register<IMetric<double[]>>(new Euclidean());
        }

        public T GetService<T>()
        {
            return (T)_servicesDictionary[typeof(T)];
        }

        public PsoServiceLocator Register<T>(T service)
        {
            try
            {
                _servicesDictionary.Add(typeof (T), service);
            }
            catch (ArgumentException e)
            {
                throw new ArgumentException("Service already registered.");
            }
            return this;
        }
    }
}