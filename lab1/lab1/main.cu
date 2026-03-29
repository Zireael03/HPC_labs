#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


// CPU 
void matMulCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// GPU 
__global__ void matMulKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matMulGPU(const float* h_A, const float* h_B, float* h_C, int N) {
    size_t size = N * N * sizeof(float);

    float* d_A, * d_B, * d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
        (N + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matMulKernel << <gridDim, blockDim >> > (d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

bool verifyResults(const float* CPU, const float* GPU, int N, float epsilon = 1e-2f) {
    for (int i = 0; i < N * N; ++i) {
        if (std::abs(CPU[i] - GPU[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

void initMatrix(float* mat, int N) {
    for (int i = 0; i < N * N; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    std::vector<int> sizes;
    for (int n = 100; n <= 2000; n += 200) {
        sizes.push_back(n);
    }
    sizes.push_back(2000);

    std::cout << "=================================================================" << std::endl;
    std::cout << "Matrix Multiplication Benchmark (CPU vs GPU)" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << std::setw(10) << "Size"
        << std::setw(15) << "CPU Time (ms)"
        << std::setw(15) << "GPU Time (ms)"
        << std::setw(12) << "Speedup"
        << std::setw(10) << "Valid" << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;

    for (int N : sizes) {
        size_t size = N * N * sizeof(float);
        float* h_A = new float[N * N];
        float* h_B = new float[N * N];
        float* h_C_CPU = new float[N * N];
        float* h_C_GPU = new float[N * N];

        initMatrix(h_A, N);
        initMatrix(h_B, N);

        auto start_cpu = std::chrono::high_resolution_clock::now();
        matMulCPU(h_A, h_B, h_C_CPU, N);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> time_cpu = end_cpu - start_cpu;

        float* d_A, * d_B, * d_C;
        CUDA_CHECK(cudaMalloc((void**)&d_A, size));
        CUDA_CHECK(cudaMalloc((void**)&d_B, size));
        CUDA_CHECK(cudaMalloc((void**)&d_C, size));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        matMulKernel << <gridDim, blockDim >> > (d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float gpu_time_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));

        CUDA_CHECK(cudaMemcpy(h_C_GPU, d_C, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        bool valid = verifyResults(h_C_CPU, h_C_GPU, N);

        float speedup = time_cpu.count() / gpu_time_ms;

        std::cout << std::setw(10) << N << "x" << N
            << std::setw(15) << std::fixed << std::setprecision(2) << time_cpu.count()
            << std::setw(15) << std::fixed << std::setprecision(2) << gpu_time_ms
            << std::setw(12) << std::fixed << std::setprecision(2) << speedup
            << std::setw(10) << (valid ? "YES" : "NO") << std::endl;

        delete[] h_A;
        delete[] h_B;
        delete[] h_C_CPU;
        delete[] h_C_GPU;
    }

    std::cout << "=================================================================" << std::endl;
    return 0;
}