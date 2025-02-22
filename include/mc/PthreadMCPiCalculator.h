#pragma once

#include <pthread.h>

#include <vector>

#include "mc/MCPiCalculator.h"
#include "random/RandomBuilder.hpp"

struct PthreadTaskData {
    RNGType            rngType;
    DistType           distType;
    unsigned long long chunkSize;
    unsigned long long chunkCount;
    unsigned long long insideCount;
    unsigned int       seed;
};

void* pthreadTask(void* arg);

class PthreadMCPiCalculator : public IMCPiCalculator {
   public:
    long double estimatePi(unsigned long long totalSamples, int threadCount,
                      unsigned long long chunkSize, RNGType rngType, DistType distType) override;
};

IMCPiCalculator* createPthreadCalculator();