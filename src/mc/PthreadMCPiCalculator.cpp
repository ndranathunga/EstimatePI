#include "mc/PthreadMCPiCalculator.h"

void* pthreadTask(void* arg) {
    PthreadTaskData* data = static_cast<PthreadTaskData*>(arg);
    data->insideCount     = 0;
    Random<RNGType::MT19937, DistType::UniformReal> random(data->seed, -1.0, 1.0);
    for (long long i = 0; i < data->chunkSize; ++i) {
        double x = random.next();
        double y = random.next();
        if (x * x + y * y <= 1.0) {
            data->insideCount++;
        }
    }
    return nullptr;
}

double PthreadMCPiCalculator::estimatePi(long long totalSamples, int threadCount,
                                         long long chunkSize, RNGType rngType = RNGType::MT19937,
                                         DistType distType = DistType::UniformReal) {
    // FIXME: ChatGPT generated code
    int                          numChunks        = totalSamples / chunkSize;
    long long                    totalInsideCount = 0;
    std::vector<pthread_t>       threads(threadCount);
    std::vector<PthreadTaskData> taskData(threadCount);
    int                          chunkIndex = 0;
    while (chunkIndex < numChunks) {
        int activeThreads = std::min(threadCount, numChunks - chunkIndex);
        for (int i = 0; i < activeThreads; ++i) {
            taskData[i].chunkSize = chunkSize;
            taskData[i].seed      = static_cast<unsigned int>(chunkIndex * 100 + i + 42);
            pthread_create(&threads[i], nullptr, pthreadTask, &taskData[i]);
            chunkIndex++;
        }
        for (int i = 0; i < activeThreads; ++i) {
            pthread_join(threads[i], nullptr);
            totalInsideCount += taskData[i].insideCount;
        }
    }
    return 4.0 * totalInsideCount / totalSamples;
}

IMCPiCalculator* createPthreadCalculator() {
    return new PthreadMCPiCalculator();
}