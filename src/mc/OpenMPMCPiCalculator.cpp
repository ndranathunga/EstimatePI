#include "mc/OpenMPMCPiCalculator.h"

static long long simulateChunk_OpenMP(long long chunkSamples, RNGType rngType, DistType distType) {
    long long          insideCount = 0;
    unsigned int       threadSeed  = omp_get_thread_num() + 42;  // Unique seed per thread
    std::random_device rd;
    IRandom* random = RandomBuilder::build(rngType, distType, rd() + threadSeed, -1.0, 1.0);

    for (long long sampleIndex = 0; sampleIndex < chunkSamples; ++sampleIndex) {
        double x = random->next();
        double y = random->next();
        if (x * x + y * y <= 1.0) {
            insideCount++;
        }
    }
    return insideCount;
}

double OpenMPMCPiCalculator::estimatePi(long long totalSamples, int threadCount,
                                        long long chunkSize, RNGType rngType, DistType distType) {
    long long chunkCount       = totalSamples / chunkSize;
    long long totalInsideCount = 0;

#pragma omp parallel for num_threads(threadCount) reduction(+ : totalInsideCount)
    for (long long chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
        totalInsideCount += simulateChunk_OpenMP(chunkSize, rngType, distType);
    }

    return 4.0 * totalInsideCount / totalSamples;
}

IMCPiCalculator* createOpenMPCalculator() {
    return new OpenMPMCPiCalculator();
}