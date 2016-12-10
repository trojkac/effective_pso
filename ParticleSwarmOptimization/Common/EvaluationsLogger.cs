using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common
{
    public class EvaluationsLogger
    {
        public int _gpuEvaluations { get; private set; }
        public int _cpuEvaluations { get; private set; }

        public void IncreaseGpuEvals(int newEvals)
        {
            _gpuEvaluations += newEvals;
        }

        public void IncreaseCpuEvals(int newEvals)
        {
            _cpuEvaluations += newEvals;
        }

        public void RestartCounters()
        {
            _gpuEvaluations = 0;
            _cpuEvaluations = 0;
        }



        public double Ratio
        {
            get
            {
                return _gpuEvaluations / ((double)_cpuEvaluations);
            }
        }
    }
}
