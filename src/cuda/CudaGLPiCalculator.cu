#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "gl/CudaGLPiCalculator.h"

#define CUDA_CHECK(call)                                                                 \
    {                                                                                    \
        cudaError_t err = call;                                                          \
        if (err != cudaSuccess) {                                                        \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                           \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }

// ----------------------------------------------------
// Atomic Add for double if arch < SM_60
// ----------------------------------------------------
#if __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int  old            = *address_as_ull, assumed;
    do {
        assumed                     = old;
        unsigned long long int next = __double_as_longlong(__longlong_as_double(assumed) + val);
        old                         = atomicCAS(address_as_ull, assumed, next);
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ inline double atomicAddDouble(double* address, double val) {
    return atomicAdd(address, val);
}
#endif

__global__ void gregoryLeibnizKernel(double* d_globalSum, unsigned long long startTerm,
                                     unsigned long long endTerm) {
    unsigned long long idx    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = blockDim.x * gridDim.x;

    double localSum = 0.0f64;

    for (unsigned long long n = startTerm + idx; n <= endTerm; n += stride) {
        double sign  = ((n & 1ULL) == 0ULL) ? 1.0 : -1.0;
        double denom = 2.0 * static_cast<double>(n) + 1.0;
        localSum += (sign / denom);
    }

    atomicAddDouble(d_globalSum, localSum);
}

long double CudaGLPiCalculator::estimatePi(unsigned long long maxTerms) {
    if (maxTerms == 0) {
        throw std::invalid_argument("maxTerms must be > 0");
    }

    long double sum        = partialSumOnDevice(0, maxTerms - 1);
    long double piEstimate = static_cast<long double>(4.0f64 * sum);
    return piEstimate;
}

long double CudaGLPiCalculator::partialSumOnDevice(unsigned long long startTerm,
                                                   unsigned long long endTerm) {
    if (endTerm < startTerm) {
        return 0.0;
    }

    double h_sum = 0.0;

    double* d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_sum, &h_sum, sizeof(double), cudaMemcpyHostToDevice));

    const int          threadsPerBlock = 1024;
    unsigned long long totalTerms      = (endTerm - startTerm + 1ULL);
    int                blocks          = (totalTerms + threadsPerBlock - 1) / threadsPerBlock;

    gregoryLeibnizKernel<<<blocks, threadsPerBlock>>>(d_sum, startTerm, endTerm);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sum));

    return h_sum;
}

long double CudaGLPiCalculator::estimatePiPrecision(unsigned int decimalPlaces) {
    if (decimalPlaces == 0) {
        throw std::invalid_argument("decimalPlaces must be > 0");
    }
    long double tolerance = powl(10.0L, -(static_cast<long double>(decimalPlaces) + 1.0L));

    unsigned long long chunkSize      = 100000ULL;  // 1e5 per chunk
    unsigned long long currentStart   = 0;
    long double        accumulatedSum = 0.0L;  // partial sum of series
    long double        oldValue       = 0.0L;
    bool               done           = false;

    while (!done) {
        unsigned long long currentEnd = currentStart + chunkSize - 1ULL;
        // Summation on device for [currentStart, currentEnd]
        double partialSum = partialSumOnDevice(currentStart, currentEnd);

        accumulatedSum += static_cast<long double>(partialSum);
        long double piEstimate = 4.0L * accumulatedSum;

        if (fabsl(piEstimate - (4.0L * oldValue)) < tolerance) {
            done = true;
        } else {
            oldValue = accumulatedSum;
            currentStart += chunkSize;
        }
    }

    return 4.0L * accumulatedSum;
}

CudaGLPiCalculator* createCudaGLPiCalculator() {
    return new CudaGLPiCalculator();
}
