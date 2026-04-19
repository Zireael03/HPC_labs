#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

// Константы алгоритма
const float ALPHA = 0.04f;

// Гауссовы веса в constant memory
__constant__ float d_GAUSS_WEIGHTS[9];

const float h_GAUSS_WEIGHTS[9] = {
    0.0751f, 0.1238f, 0.0751f,
    0.1238f, 0.2042f, 0.1238f,
    0.0751f, 0.1238f, 0.0751f
};

// ---------------------- CUDA KERNELS ----------------------
__global__ void computeHarrisResponseKernel(float* d_R, int width, int height, float alpha, float threshold, cudaTextureObject_t texObj) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Чтение окна 3x3 через текстуру с hardware clamping
    float I[3][3];
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            I[i + 1][j + 1] = tex2D<float>(texObj, x + j, y + i);
        }
    }

    float Sxx = 0.0f, Syy = 0.0f, Sxy = 0.0f;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float Ix = (I[i][min(j + 1, 2)] - I[i][max(j - 1, 0)]) / 2.0f;
            float Iy = (I[min(i + 1, 2)][j] - I[max(i - 1, 0)][j]) / 2.0f;

            int idx = i * 3 + j;
            Sxx += d_GAUSS_WEIGHTS[idx] * Ix * Ix;
            Syy += d_GAUSS_WEIGHTS[idx] * Iy * Iy;
            Sxy += d_GAUSS_WEIGHTS[idx] * Ix * Iy;
        }
    }

    float det = Sxx * Syy - Sxy * Sxy;
    float trace = Sxx + Syy;
    float R = det - alpha * trace * trace;

    d_R[y * width + x] = (R > threshold) ? R : 0.0f;
}

__global__ void nonMaxSuppressionKernel(const float* d_R, int* d_corners, int width, int height, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float R = d_R[y * width + x];
    if (R <= threshold) {
        d_corners[y * width + x] = 0;
        return;
    }

    bool isMax = true;
    for (int i = -1; i <= 1 && isMax; ++i) {
        for (int j = -1; j <= 1 && isMax; ++j) {
            if (i == 0 && j == 0) continue;
            int nx = max(0, min(x + j, width - 1));
            int ny = max(0, min(y + i, height - 1));
            if (d_R[ny * width + nx] > R) isMax = false;
        }
    }
    d_corners[y * width + x] = isMax ? 1 : 0;
}

// ---------------------- CPU РЕАЛИЗАЦИЯ ----------------------
void computeHarrisCPU(const Mat& img, float threshold, Mat& outCorners) {
    int h = img.rows, w = img.cols;
    outCorners = Mat::zeros(h, w, CV_8UC1);
    vector<float> R(h * w, 0.0f);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float I[3][3];
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int ny = max(0, min(y + i, h - 1));
                    int nx = max(0, min(x + j, w - 1));
                    I[i + 1][j + 1] = img.at<float>(ny, nx);
                }
            }

            float Sxx = 0, Syy = 0, Sxy = 0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    float Ix = (I[i][min(j + 1, 2)] - I[i][max(j - 1, 0)]) / 2.0f;
                    float Iy = (I[min(i + 1, 2)][j] - I[max(i - 1, 0)][j]) / 2.0f;
                    int idx = i * 3 + j;
                    Sxx += h_GAUSS_WEIGHTS[idx] * Ix * Ix;
                    Syy += h_GAUSS_WEIGHTS[idx] * Iy * Iy;
                    Sxy += h_GAUSS_WEIGHTS[idx] * Ix * Iy;
                }
            }

            float det = Sxx * Syy - Sxy * Sxy;
            float trace = Sxx + Syy;
            R[y * w + x] = det - ALPHA * trace * trace;
        }
    }

    // NMS
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (R[y * w + x] <= threshold) continue;
            bool isMax = true;
            for (int i = -1; i <= 1 && isMax; ++i) {
                for (int j = -1; j <= 1 && isMax; ++j) {
                    if (i == 0 && j == 0) continue;
                    int ny = max(0, min(y + i, h - 1));
                    int nx = max(0, min(x + j, w - 1));
                    if (R[ny * w + nx] > R[y * w + x]) isMax = false;
                }
            }
            if (isMax) outCorners.at<uchar>(y, x) = 255;
        }
    }
}

// ---------------------- НАСТРОЙКА ТЕКСТУРЫ ----------------------
cudaError_t setupTexture(const float* d_data, int width, int height, cudaTextureObject_t* pTexObject) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    resDesc.res.pitch2D.devPtr = (void*)d_data;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = width * sizeof(float);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;  // Hardware clamping (требование PDF!)
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    return cudaCreateTextureObject(pTexObject, &resDesc, &texDesc, nullptr);
}

void destroyTexture(cudaTextureObject_t texObject) {
    if (texObject != 0) {
        cudaDestroyTextureObject(texObject);
    }
}

// ---------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------------------
void drawCorners(Mat& img, const Mat& corners) {
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            if (corners.at<uchar>(y, x) == 255) {
                circle(img, Point(x, y), 3, Scalar(0, 0, 255), -1);
            }
        }
    }
}

int countCorners(const Mat& corners) {
    return countNonZero(corners);
}

// ---------------------- MAIN ----------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <input_image_path> <threshold>\n";
        cout << "Example: harris_lab.exe image.png 1e7\n";
        return 1;
    }

    string imgPath = argv[1];
    float threshold = stof(argv[2]);

    Mat src = imread(imgPath, IMREAD_GRAYSCALE);
    if (src.empty()) { cerr << "Error: Cannot load image\n"; return 1; }

    int h = src.rows, w = src.cols;
    Mat srcFloat;
    src.convertTo(srcFloat, CV_32FC1);

    // Копируем веса в constant memory
    cudaMemcpyToSymbol(d_GAUSS_WEIGHTS, h_GAUSS_WEIGHTS, 9 * sizeof(float));

    // --- CPU ВЕРСИЯ ---
    Mat cpuCorners;
    auto cpuStart = chrono::high_resolution_clock::now();
    computeHarrisCPU(srcFloat, threshold, cpuCorners);
    auto cpuEnd = chrono::high_resolution_clock::now();
    double cpuTime = chrono::duration_cast<chrono::milliseconds>(cpuEnd - cpuStart).count();

    // --- GPU ВЕРСИЯ ---
    float* d_image = nullptr;
    float* d_R = nullptr;
    int* d_corners = nullptr;
    cudaTextureObject_t texObj = 0;

    cudaMalloc(&d_image, w * h * sizeof(float));
    cudaMalloc(&d_R, w * h * sizeof(float));
    cudaMalloc(&d_corners, w * h * sizeof(int));

    cudaMemcpy(d_image, srcFloat.ptr<float>(), w * h * sizeof(float), cudaMemcpyHostToDevice);

    // Создаем объект текстуры (ТРЕБОВАНИЕ PDF 4.3)
    cudaError_t err = setupTexture(d_image, w, h, &texObj);
    if (err != cudaSuccess) {
        cerr << "Error creating texture object: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    auto gpuStart = chrono::high_resolution_clock::now();
    // Передаем texObj как параметр kernel
    computeHarrisResponseKernel << <grid, block >> > (d_R, w, h, ALPHA, threshold, texObj);
    cudaDeviceSynchronize();
    nonMaxSuppressionKernel << <grid, block >> > (d_R, d_corners, w, h, threshold);
    cudaDeviceSynchronize();
    auto gpuEnd = chrono::high_resolution_clock::now();
    double gpuTime = chrono::duration_cast<chrono::milliseconds>(gpuEnd - gpuStart).count();

    Mat gpuCorners(h, w, CV_8UC1);
    vector<int> h_corners(w * h);
    cudaMemcpy(h_corners.data(), d_corners, w * h * sizeof(int), cudaMemcpyDeviceToHost);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            gpuCorners.at<uchar>(y, x) = h_corners[y * w + x] ? 255 : 0;

    // --- СРАВНЕНИЕ ---
    int cpuCount = countCorners(cpuCorners);
    int gpuCount = countCorners(gpuCorners);
    int matching = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (cpuCorners.at<uchar>(y, x) == gpuCorners.at<uchar>(y, x)) matching++;

    double matchPercent = (100.0 * matching) / (w * h);
    cout << "=== REPORT ===\n";
    cout << "CPU Time: " << cpuTime << " ms | Detected: " << cpuCount << " corners\n";
    cout << "GPU Time: " << gpuTime << " ms | Detected: " << gpuCount << " corners\n";
    cout << "Pixel match: " << matchPercent << "%\n";
    cout << "Result: " << (matchPercent > 99.0 ? "COINCIDENCE" : "MISMATCH") << "\n\n";

    // --- СОХРАНЕНИЕ ---
    Mat outCPU, outGPU;
    cvtColor(src, outCPU, COLOR_GRAY2BGR);
    cvtColor(src, outGPU, COLOR_GRAY2BGR);
    drawCorners(outCPU, cpuCorners);
    drawCorners(outGPU, gpuCorners);

    imwrite("output_cpu.png", outCPU);
    imwrite("output_gpu.png", outGPU);
    cout << "Images saved: output_cpu.png, output_gpu.png\n";

    // Очистка
    destroyTexture(texObj);
    cudaFree(d_image);
    cudaFree(d_R);
    cudaFree(d_corners);

    return 0;
}