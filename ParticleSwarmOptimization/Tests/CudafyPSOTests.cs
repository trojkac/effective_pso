using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using PSOCudafy;

namespace Tests
{
    [TestClass]
    public class CudafyPSOTests
    {
        [TestMethod]
        public void SimpleTest()
        {
            PSOCudafy.PsoAlgorithm.Execute();
        }
    }
}
