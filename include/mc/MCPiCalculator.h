#pragma once

#include "random/Random.hpp"

class IMCPiCalculator {
   public:
    virtual ~IMCPiCalculator() {
    }
    virtual long double estimatePi(unsigned long long totalSamples, int threadCount,
                              unsigned long long chunkSize, RNGType rngType = RNGType::MT19937,
                              DistType distType = DistType::UniformReal) = 0;
};
