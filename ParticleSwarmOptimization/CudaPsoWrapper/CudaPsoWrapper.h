// CudaPsoWrapper.h

#pragma once

#include <tuple>
#include <vector>
#include <limits>

using namespace System;

namespace CudaPsoWrapper {
	public ref class CudaPSOAlgorithm
	{
    private:
        std::tuple<std::vector<double>, double>* localEndpoint;

        std::tuple<std::vector<double>, double>* remoteEndpoint;

        int iterationsCount;

    public:
        static CudaPSOAlgorithm^ createAlgorithm(int iterations)
        {
            auto alg = gcnew CudaPSOAlgorithm;

            alg->localEndpoint = new std::tuple<std::vector<double>, double>(std::vector<double>(), std::numeric_limits<double>::max());
            alg->remoteEndpoint = new std::tuple<std::vector<double>, double>(std::vector<double>(), std::numeric_limits<double>::max());
            alg->iterationsCount = iterations;

            return alg;
        };

        void run();

        void* getLocalEndpoint()
        {
            return localEndpoint;
        }

        void* getRemoteEndpoint()
        {
            return remoteEndpoint;
        }
	};
}
