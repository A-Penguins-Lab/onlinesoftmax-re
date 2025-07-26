#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <utility>
#include <math.h>

using namespace std;

__global__ void softmax(float* vecA, float* sum, int N) {
    int idx = threadIdx.x;
    if (idx < N) {
        vecA[idx] = vecA[idx] / (*sum);  // dereferencing sum properly
    }
}

__global__ void ExponentiateKernel(float* vecA, int N) {
    int idx = threadIdx.x;
    if (idx < N) {
        vecA[idx] = expf(vecA[idx]);
    }
}

__global__ void SumReduction(float* vecA, float* sum, int N) {
	unsigned int idx = threadIdx.x;

	for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2){
		if (threadIdx.x < stride) {
			vecA[idx] += vecA[idx + stride];
		}
	}

	if (threadIdx.x == 0){
		*sum = vec[threadIdx.x];
	}
}

float* randomVec(int N) {
    float* M = new float[N];

    for (int i = 0; i < N; i++) {
        M[i] = 0.21f * i;
    }

    return M;
}

void RunSoftMax(int N = 4096) {
    float* M = randomVec(N);

    float *Md, *sumd;
    cudaMalloc(&Md, sizeof(float) * N);
    cudaMalloc(&sumd, sizeof(float));
    cudaMemset(sumd, 0, sizeof(float));  // Important: initialize sum to 0

    cudaMemcpy(Md, M, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Launch kernels
    ExponentiateKernel<<<1, N>>>(Md, N);
    SumReduction<<<1, N>>>(Md, sumd, N);

    float sum;
    cudaMemcpy(&sum, sumd, sizeof(float), cudaMemcpyDeviceToHost);

    softmax<<<1, N>>>(Md, &sum, N);

    float* result = new float[N];
    cudaMemcpy(result, Md, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Debug print
    for (int i = 0; i < N; i++) {
        cout << result[i] << " ";
    }
    cout << endl;

    // Clean up
    cudaFree(Md);
    cudaFree(sumd);

    delete[] M;
    delete[] result;

    cout << "Softmax done and so are all allocs and frees" << endl;
}

int main() {
    int N = 128;
    RunSoftMax(N);
    return 0;
}

