#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <windows.h>

const int BLOCK_SIZE = 16;
constexpr float EPSILON = 1e-10f;

cudaTextureObject_t g_textureRef = 0;

void saveGrayscaleToBMP(float* imageData, int imgWidth, int imgHeight, const char* filename) {
    unsigned char* buffer = new unsigned char[imgWidth * imgHeight];

    for (int idx = 0; idx < imgWidth * imgHeight; ++idx) {
        float clampedValue = fmaxf(0.0f, fminf(255.0f, imageData[idx]));
        buffer[idx] = static_cast<unsigned char>(clampedValue);
    }

    stbi_write_bmp(filename, imgWidth, imgHeight, 1, buffer);
    delete[] buffer;
}

__host__ __device__ inline float computeSpatialWeight(int dx, int dy, float sigmaSpace) {
    float distSq = static_cast<float>(dx * dx + dy * dy);
    return expf(-distSq / (2.0f * sigmaSpace * sigmaSpace)); 
}

__host__ __device__ inline float computeRangeWeight(float intensityDiff, float sigmaRange) {
    return expf(-(intensityDiff * intensityDiff) / (2.0f * sigmaRange * sigmaRange)); 
}

void applyBilateralFilterCPU(const float* inputImg, float* outputImg,
    int height, int width,
    float sigmaSpace, float sigmaRange) {

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            float centerPixel = inputImg[row * width + col];
            float weightSum = 0.0f;
            float weightedSum = 0.0f;

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int neighborRow = row + dy;
                    int neighborCol = col + dx;

                    neighborRow = max(0, min(neighborRow, height - 1));
                    neighborCol = max(0, min(neighborCol, width - 1));

                    float neighborPixel = inputImg[neighborRow * width + neighborCol];

                    float spatialWeight = computeSpatialWeight(dx, dy, sigmaSpace);
                    float rangeWeight = computeRangeWeight(neighborPixel - centerPixel, sigmaRange);
                    float totalWeight = spatialWeight * rangeWeight;

                    weightSum += totalWeight;
                    weightedSum += neighborPixel * totalWeight;
                }
            }

            outputImg[row * width + col] = weightedSum / (weightSum + EPSILON);
        }
    }
}

__global__ void applyBilateralFilterGPU(float* outputImage, int imgHeight, int imgWidth,
    float sigmaSpace, float sigmaRange,
    cudaTextureObject_t textureData) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= imgHeight || col >= imgWidth) return;

    float centerValue = tex2D<float>(textureData, col + 0.5f, row + 0.5f);
    float accumulator = 0.0f;
    float weightTotal = 0.0f;


#pragma unroll
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
      
            float neighborValue = tex2D<float>(textureData,
                col + offsetX + 0.5f,
                row + offsetY + 0.5f);

            float wSpatial = computeSpatialWeight(offsetX, offsetY, sigmaSpace);
            float wRange = computeRangeWeight(neighborValue - centerValue, sigmaRange);
            float combinedWeight = wSpatial * wRange;

            accumulator += neighborValue * combinedWeight;
            weightTotal += combinedWeight;
        }
    }

    outputImage[row * imgWidth + col] = accumulator / (weightTotal + EPSILON);
}

cudaError_t initTextureObject(const float* deviceData, int width, int height,
    cudaTextureObject_t* textureObj) {
    cudaResourceDesc resourceDesc;
    memset(&resourceDesc, 0, sizeof(resourceDesc));
    resourceDesc.resType = cudaResourceTypePitch2D;
    resourceDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    resourceDesc.res.pitch2D.devPtr = const_cast<float*>(deviceData);
    resourceDesc.res.pitch2D.width = width;
    resourceDesc.res.pitch2D.height = height;
    resourceDesc.res.pitch2D.pitchInBytes = width * sizeof(float);

    cudaTextureDesc textureDesc;
    memset(&textureDesc, 0, sizeof(textureDesc));
    textureDesc.addressMode[0] = cudaAddressModeClamp;  
    textureDesc.addressMode[1] = cudaAddressModeClamp;
    textureDesc.filterMode = cudaFilterModePoint;         
    textureDesc.readMode = cudaReadModeElementType;
    textureDesc.normalizedCoords = 0;                    

    return cudaCreateTextureObject(textureObj, &resourceDesc, &textureDesc, nullptr);
}

void releaseTextureObject(cudaTextureObject_t textureObj) {
    if (textureObj != 0) {
        cudaDestroyTextureObject(textureObj);
    }
}

int main(int argc, char** argv) {
    SetConsoleOutputCP(1251);
    int imgWidth, imgHeight, channels;
    unsigned char* rawData = stbi_load("image_8k.bmp", &imgWidth, &imgHeight, &channels, 1);

    if (!rawData) {
        fprintf(stderr, "ERROR: Не удалось загрузить image_8k.bmp\n");
        fprintf(stderr, "Убедитесь, что файл находится в папке с программой\n");
        return -1;
    }

    printf("===========================================\n");
    printf("Bilateral Filter - CUDA Implementation\n");
    printf("===========================================\n");
    printf("Image dimensions: %d x %d pixels\n", imgWidth, imgHeight);
    printf("Total pixels: %.2f MP\n", (imgWidth * imgHeight) / 1e6);

    float* h_inputImage = new float[imgWidth * imgHeight];
    float* h_outputCPU = new float[imgWidth * imgHeight];
    float* h_outputGPU = new float[imgWidth * imgHeight];

    for (int i = 0; i < imgWidth * imgHeight; ++i) {
        h_inputImage[i] = static_cast<float>(rawData[i]);
    }
    stbi_image_free(rawData);

    float sigmaSpace, sigmaRange;
    std::cout << "\nВведите sigma_d (пространственный параметр): ";
    std::cin >> sigmaSpace;
    std::cout << "Введите sigma_r (параметр яркости): ";
    std::cin >> sigmaRange;

    printf("\n[CPU] Запуск билатеральной фильтрации...\n");
    clock_t cpuStart = clock();

    applyBilateralFilterCPU(h_inputImage, h_outputCPU,
        imgHeight, imgWidth,
        sigmaSpace, sigmaRange);

    clock_t cpuEnd = clock();
    float cpuTimeSec = static_cast<float>(cpuEnd - cpuStart) / CLOCKS_PER_SEC;

    printf("[CPU] Время выполнения: %.6f сек\n", cpuTimeSec);

    saveGrayscaleToBMP(h_outputCPU, imgWidth, imgHeight, "output_8k_cpu.bmp");
    printf("[CPU] Результат сохранен в outpu_8k_cpu.bmp\n");

    printf("\n[GPU] Подготовка данных...\n");

    float* d_input = nullptr, * d_output = nullptr;
    size_t imageSize = imgWidth * imgHeight * sizeof(float);

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    cudaMemcpy(d_input, h_inputImage, imageSize, cudaMemcpyHostToDevice);

    cudaTextureObject_t texObj = 0;
    cudaError_t err = initTextureObject(d_input, imgWidth, imgHeight, &texObj);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: Создание текстуры не удалось: %s\n",
            cudaGetErrorString(err));
        return -1;
    }

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((imgWidth + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (imgHeight + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    printf("[GPU] Запуск kernel...\n");
    cudaEventRecord(startEvent, 0);

    applyBilateralFilterGPU << <gridDim, blockDim >> > (d_output, imgHeight, imgWidth,
        sigmaSpace, sigmaRange, texObj);

    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        fprintf(stderr, "ERROR: Запуск kernel не удался: %s\n",
            cudaGetErrorString(kernelErr));
        return -1;
    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuKernelTimeMs;
    cudaEventElapsedTime(&gpuKernelTimeMs, startEvent, stopEvent);

    cudaEventRecord(startEvent, 0);
    cudaMemcpy(h_outputGPU, d_output, imageSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuMemcpyTimeMs;
    cudaEventElapsedTime(&gpuMemcpyTimeMs, startEvent, stopEvent);

    float totalGpuTimeSec = (gpuKernelTimeMs + gpuMemcpyTimeMs) / 1000.0f;

    printf("[GPU] Время фильтрации: %.6f сек\n", gpuKernelTimeMs / 1000.0f);
    printf("[GPU] Время копирования: %.6f сек\n", gpuMemcpyTimeMs / 1000.0f);
    printf("[GPU] Общее время: %.6f сек\n", totalGpuTimeSec);

    saveGrayscaleToBMP(h_outputGPU, imgWidth, imgHeight, "output_8k_gpu.bmp");
    printf("[GPU] Результат сохранен в output_8k_gpu.bmp\n");

    printf("\n===========================================\n");
    printf("РЕЗУЛЬТАТЫ СРАВНЕНИЯ\n");
    printf("===========================================\n");
    printf("Ускорение (включая memcpy): %.2f x\n", cpuTimeSec / totalGpuTimeSec);
    printf("Ускорение (только kernel):  %.2f x\n",
        cpuTimeSec / (gpuKernelTimeMs / 1000.0f));
    printf("===========================================\n");

    releaseTextureObject(texObj);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    delete[] h_inputImage;
    delete[] h_outputCPU;
    delete[] h_outputGPU;

    printf("\nПрограмма завершена успешно.\n");
    return 0;
}