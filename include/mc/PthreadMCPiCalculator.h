#pragma once

#include <pthread.h>

#include <vector>

#include "mc/MCPiCalculator.h"
#include "random/Random.hpp"

struct PthreadTaskData {
    long long    chunkSize;
    long long    insideCount;
    unsigned int seed;
};

void* pthreadTask(void* arg);

class PthreadMCPiCalculator : public IMCPiCalculator {
   public:
    double estimatePi(long long totalSamples, int threadCount, long long chunkSize, RNGType rngType,
                      DistType distType) override;
};

IMCPiCalculator* createPthreadCalculator();