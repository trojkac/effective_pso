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
          where T: class
        {
          object service;
          if (_servicesDictionary.TryGetValue(typeof(T), out service))
          {
            return (T)service;
          }
          return null;
        }

        public PsoServiceLocator Register<T>(T service)
          where T: class 
        {
            try
            {
                _servicesDictionary.Add(typeof (T), service);
            }
            catch (ArgumentException)
            {
                throw new ArgumentException("Service already registered.");
            }
            return this;
        }
    }
}