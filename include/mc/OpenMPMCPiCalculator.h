#pragma once

#include <omp.h>

#include "mc/MCPiCalculator.h"
#include "random/RandomBuilder.hpp"

static long long simulateChunk_OpenMP(long long chunkSamples, RNGType rngType, DistType distType);

class OpenMPMCPiCalculator : public IMCPiCalculator {
   public:
    double estimatePi(long long totalSamples, int threadCount, long long chunkSize, RNGType rngType,
                      DistType distType) override;
};

IMCPiCalculator* createOpenMPCalculator();
