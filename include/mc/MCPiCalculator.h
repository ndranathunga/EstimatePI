#pragma once

#include "random/Random.hpp"

class IMCPiCalculator {
   public:
    virtual ~IMCPiCalculator() {
    }
    virtual double estimatePi(long long totalSamples, int threadCount, long long chunkSize,
                              RNGType  rngType  = RNGType::MT19937,
                              DistType distType = DistType::UniformReal) = 0;
};
