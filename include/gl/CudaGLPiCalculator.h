#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "GLPiCalculator.h"

class CudaGLPiCalculator : public IGLPiCalculator {
   public:
    CudaGLPiCalculator()  = default;
    ~CudaGLPiCalculator() = default;

    long double estimatePi(unsigned long long maxTerms) override;
    long double estimatePiPrecision(unsigned int decimalPlaces) override;

   private:
    long double partialSumOnDevice(unsigned long long startTerm, unsigned long long endTerm);
};

CudaGLPiCalculator* createCudaGLPiCalculator();
