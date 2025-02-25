#include "mc/PthreadMCPiCalculator.h"

void* pthreadTask(void* arg) {
    PthreadTaskData* data = static_cast<PthreadTaskData*>(arg);

    unsigned long long insideCount = 0;
    unsigned long long chunkCount  = data->chunkCount;
    unsigned long long chunkSize   = data->chunkSize;

    std::random_device rd;
    IRandom*           random =
        RandomBuilder::build(data->rngType, data->distType, rd() + data->seed, -1.0, 1.0);

    for (unsigned long long chunk = 0; chunk < chunkCount; ++chunk) {
        for (unsigned long long i = 0; i < chunkSize; ++i) {
            double x = random->next();
            double y = random->next();
            if (x * x + y * y <= 1.0) {
                insideCount++;
            }
        }
    }
    data->insideCount = insideCount;

    return nullptr;
}

long double PthreadMCPiCalculator::estimatePi(unsigned long long totalSamples, int threadCount,
                                              unsigned long long chunkSize, RNGType rngType,
                                              DistType distType) {
    int                          totalChunks      = totalSamples / chunkSize;
    unsigned long long           totalInsideCount = 0;
    std::vector<pthread_t>       threads(threadCount);
    std::vector<PthreadTaskData> taskData(threadCount);
    int                          chunkIndex = 0;

    int chunksPerThread = totalChunks / threadCount;
    int remainder       = totalChunks % threadCount;

    for (int i = 0; i < threadCount; ++i) {
        taskData[i].rngType    = rngType;
        taskData[i].distType   = distType;
        taskData[i].chunkSize  = chunkSize;
        taskData[i].seed       = static_cast<unsigned int>(i + 42);
        taskData[i].chunkCount = chunksPerThread + (i < remainder ? 1 : 0);
        pthread_create(&threads[i], nullptr, pthreadTask, &taskData[i]);
    }

    for (int i = 0; i < threadCount; ++i) {
        pthread_join(threads[i], nullptr);
        totalInsideCount += taskData[i].insideCount;
    }

    return (long double)4.0 * totalInsideCount / totalSamples;
}

IMCPiCalculator* createPthreadCalculator() {
    return new PthreadMCPiCalculator();
}