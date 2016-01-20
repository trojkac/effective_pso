using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CudaPsoWrapper;

namespace Tests
{
    [TestClass]
    public class CudaWrapperTests
    {
        [TestMethod]
        public void SampleRun()
        {
            var algorithm = CudaPSOAlgorithm.createAlgorithm(100);
            algorithm.run();
        }
    }
}
