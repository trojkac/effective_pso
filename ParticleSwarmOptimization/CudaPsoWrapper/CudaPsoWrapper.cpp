// This is the main DLL file.

#include "CudaPsoWrapper.h"
#include "CudaCalls.h"

namespace CudaPsoWrapper
{
    void CudaPSOAlgorithm::run()
    {
        runCuda(localEndpoint, remoteEndpoint, iterationsCount, 100, 1, 10);
    }
}