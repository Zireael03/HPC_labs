#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void vectorSumKernel(const float* d_in, float* d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float threadSum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        threadSum += d_in[i];
    }
    atomicAdd(d_out, threadSum);
}

float sumVectorCPU(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (size_t i = 0; i < vec.size(); ++i) {
        sum += vec[i];
    }
    return sum;
}

float sumVectorGPU(const std::vector<float>& h_vec, double& gpuTimeMs) {
    int n = static_cast<int>(h_vec.size());
    float* d_in = nullptr;
    float* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize(); 
    cudaEventRecord(start);
    vectorSumKernel << <numBlocks, blockSize >> > (d_in, d_out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    gpuTimeMs = milliseconds;

    float h_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return h_sum;
}

int main() {
    srand(42);

    double warmupTime = 0.0;
    sumVectorGPU(std::vector<float>(1024, 1.0f), warmupTime);

    std::vector<int> sizes = { 1000, 10000, 100000, 500000, 1000000, 2000000, 4000000, 8000000 };

    std::cout << std::left << std::setw(10) << "Size"
        << std::setw(14) << "CPU (ms)"
        << std::setw(14) << "GPU (ms)"
        << std::setw(12) << "Speedup"
        << "Match\n";
    std::cout << std::string(60, '-') << "\n";

    for (int n : sizes) {
        std::vector<float> h_vec(n);
        for (int i = 0; i < n; ++i) {
            h_vec[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
        }

        auto cpuStart = std::chrono::high_resolution_clock::now();
        float cpuSum = sumVectorCPU(h_vec);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        double cpuTimeMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

        double gpuTimeMs = 0.0;
        float gpuSum = sumVectorGPU(h_vec, gpuTimeMs);

        double speedup = (gpuTimeMs > 0.001) ? cpuTimeMs / gpuTimeMs : 0.0;

        bool match = std::abs(cpuSum - gpuSum) < 1e-4f * std::abs(cpuSum);

        std::cout << std::left << std::setw(10) << n
            << std::setw(14) << std::fixed << std::setprecision(3) << cpuTimeMs
            << std::setw(14) << gpuTimeMs
            << std::setw(12) << std::setprecision(2) << speedup
            << (match ? "YES" : "NO") << "\n";
    }

    return 0;
}