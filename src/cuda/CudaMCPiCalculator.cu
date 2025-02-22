#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "mc/MCPiCalculator.h"
#include "random/Random.hpp"

__global__ void MCKernel(unsigned long long* d_inside, unsigned long long chunkSamples,
                         unsigned int seed) {
    unsigned long long localCount = 0;
    curandState        state;
    curand_init(seed, threadIdx.x + blockIdx.x * blockDim.x, 0, &state);
    for (unsigned long long i = 0; i < chunkSamples; ++i) {
        float x = curand_uniform(&state) * 2.0f - 1.0f;
        float y = curand_uniform(&state) * 2.0f - 1.0f;
        if (x * x + y * y <= 1.0f)
            localCount++;
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
        cudaMalloc(&d_inside, sizeof(unsigned long long));
        cudaMemcpy(d_inside, &h_inside, sizeof(unsigned long long), cudaMemcpyHostToDevice);

        int threadsPerBlock = 1024;
        int blocks          = (threadCount + threadsPerBlock - 1) / threadsPerBlock;

        std::random_device rd;
        unsigned int       seed = rd() + 42;
        MCKernel<<<blocks, threadsPerBlock>>>(d_inside, chunkSize, seed);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_inside, d_inside, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaFree(d_inside);
        return (long double)4.0 * h_inside / totalSamples;
    }
};

IMCPiCalculator* createCUDACalculator() {
    return new CUDAMCPiCalculator();
}
