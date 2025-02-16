#include "mc/MCFactory.h"

IMCPiCalculator* createCalculator(BackendType backend) {
    switch (backend) {
        case BackendType::OpenMP:
            return createOpenMPCalculator();
        case BackendType::Pthread:
            return createPthreadCalculator();
        case BackendType::CUDA:
            return createCUDACalculator();
        default:
            throw std::runtime_error("Unknown backend type.");
    }
}