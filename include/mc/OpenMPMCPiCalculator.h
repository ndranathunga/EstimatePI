#pragma once

#include <omp.h>

#include "mc/MCPiCalculator.h"
#include "random/RandomBuilder.hpp"

static unsigned long long simulateChunk_OpenMP(unsigned long long chunkSamples, RNGType rngType,
                                               DistType distType);

class OpenMPMCPiCalculator : public IMCPiCalculator {
   public:
    long double estimatePi(unsigned long long totalSamples, int threadCount,
                      unsigned long long chunkSize, RNGType rngType, DistType distType) override;
};

IMCPiCalculator* createOpenMPCalculator();
