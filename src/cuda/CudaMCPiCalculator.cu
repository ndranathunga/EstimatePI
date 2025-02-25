#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "mc/MCPiCalculator.h"
#include "random/Random.hpp"

#define CUDA_CHECK(call)                                                                 \
    {                                                                                    \
        cudaError_t err = call;                                                          \
        if (err != cudaSuccess) {                                                        \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                           \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }

__global__ void MCKernel(unsigned long long* d_inside, unsigned long long chunkSamples,
                         unsigned int seed, unsigned int threadCount) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= threadCount)
        return;

    unsigned long long localCount = 0;
    curandState        state;
    curand_init(seed, threadIdx.x + blockIdx.x * blockDim.x, 0, &state);
    for (unsigned long long i = 0; i < chunkSamples; ++i) {
        double x = curand_uniform_double(&state) * 2.0 - 1.0;
        double y = curand_uniform_double(&state) * 2.0 - 1.0;
        // if (x * x + y * y <= 1.0)
        //     localCount++;
        localCount += (x * x + y * y <= 1.0);
    }
    atomicAdd(d_inside, localCount);
}

class CUDAMCPiCalculator : public IMCPiCalculator {
   public:
    long double estimatePi(unsigned long long totalSamples, int threadCount,
                           unsigned long long chunkSize, RNGType rngType,
                           DistType distType) override {
        unsigned long long  h_inside = 0;
        unsigned long long* d_inside;
        CUDA_CHECK(cudaMalloc(&d_inside, sizeof(unsigned long long)));
        CUDA_CHECK(
            cudaMemcpy(d_inside, &h_inside, sizeof(unsigned long long), cudaMemcpyHostToDevice));

        int threadsPerBlock = 1024;
        int blocks          = (threadCount + threadsPerBlock - 1) / threadsPerBlock;

        std::random_device rd;
        unsigned int       seed = rd() + 42;
        MCKernel<<<blocks, threadsPerBlock>>>(d_inside, chunkSize, seed, threadCount);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(
            cudaMemcpy(&h_inside, d_inside, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_inside));

        return (long double)4.0 * h_inside / totalSamples;
    }
};

IMCPiCalculator* createCUDACalculator() {
    return new CUDAMCPiCalculator();
}
