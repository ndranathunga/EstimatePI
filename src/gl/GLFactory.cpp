#include "gl/GLFactory.h"

#include "gl/CudaGLPiCalculator.h"

IGLPiCalculator* createGLCalculator() {
    return new CudaGLPiCalculator();
}