#include <iostream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <cuda.h>

using namespace std;

constexpr int THREADDIM = 128;

__device__ __forceinline__ float maxCmp(float a, float b){
    return a > b ? a : b;
}

__global__ void ExponentialKernel(float* vecA, const float* maxv, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        vecA[idx] = expf(vecA[idx] - *maxv);
    }
}

__global__ void safeSoftmaxKernel(float* vecA, const float* sumValue, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        vecA[idx] = vecA[idx] / *sumValue;
    }
}

__global__ void sumRed(float* vecA, float* sumValue, int N) {
    __shared__ float sdata[THREADDIM];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    sdata[tid] = (idx < N) ? vecA[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *sumValue = sdata[0];  // only one block’s result is written
    }
}

__global__ void MaxReduction(float* vecA, float* maxv, int N) {
    __shared__ float sdata[THREADDIM];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    sdata[tid] = (idx < N) ? vecA[idx] : -INFINITY;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = maxCmp(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        *maxv = sdata[0];  // only one block’s result is written
    }
}

float* randomVec(int N) {
    float* M = new float[N];
    for (int i = 0; i < N; i++){
        M[i] = 0.21f * i;
    }
    return M;
}

void RunSafeSoftmax(int N) {
    float* M = randomVec(N);
    float *Md, *sumd, *maxd;

    cudaMalloc(&Md, sizeof(float) * N);
    cudaMalloc(&sumd, sizeof(float));
    cudaMalloc(&maxd, sizeof(float));

    float zero = 0.0f;
    float neg_inf = -INFINITY;
    cudaMemcpy(sumd, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(maxd, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(Md, M, sizeof(float) * N, cudaMemcpyHostToDevice);

    int threadsPerBlock = THREADDIM;
    int blocksInGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // NOTE: with your “old logic”, this only works reliably if blocksInGrid == 1
    MaxReduction<<<blocksInGrid, threadsPerBlock>>>(Md, maxd, N);
    ExponentialKernel<<<blocksInGrid, threadsPerBlock>>>(Md, maxd, N);
    sumRed<<<blocksInGrid, threadsPerBlock>>>(Md, sumd, N);
    safeSoftmaxKernel<<<blocksInGrid, threadsPerBlock>>>(Md, sumd, N);

    float* result = new float[N];
    cudaMemcpy(result, Md, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        cout << result[i] << " ";
    }
    cout << endl;

    cudaFree(Md);
    cudaFree(sumd);
    cudaFree(maxd);
    delete[] M;
    delete[] result;

    cout << "Softmax done and so are all allocs and frees" << endl;
}

int main() {
    int N = 128;
    RunSafeSoftmax(N);
    return 0;
}
