#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <utility>
#include <math.h>

using namespace std;

// Define the threadDim and blockDim from here
constexpr int THREADDIM = 128;

__global__ void softmax(float* vecA, float* sum, int N) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < N) {
		vecA[idx] = vecA[idx] / *sum;  // dereferencing sum properly
	}
}

__global__ void ExponentiateKernel(float* vecA, int N) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < N) {
		vecA[idx] = expf(vecA[idx]);
	}
}

__global__ void SumReduction(float* vecA, float* sum, int N) {
    __shared__ float sdata[THREADDIM];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    sdata[tid] = (idx < N) ? vecA[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
      *sum = sdata[0];
    }
}

__global__ void maxReduction(float* vecA, float* max, int N){ 
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float maxVal;

	for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2){
		if (threadIdx.x < stride) {
			if (vecA[idx + stride] > vecA[idx]) {
				maxVal = vecA[idx + stride];
			}
			__syncthreads();
	
		}
		__syncthreads();

	}
	
	if (threadIdx.x == 0) {
		*max = maxVal;
	}
}

float* randomVec(int N) {
	float* M = new float[N];
	
	for (int i = 0; i < N; i++) {
		M[i] = 0.21f * i;
	}

  return M;
}

void RunSoftMax(int N) {
  float* M = randomVec(N);
  float *Md, *sumd;

	cudaMalloc(&Md, sizeof(float) * N);
  cudaMalloc(&sumd, sizeof(float)); 

  cudaMemset(sumd, 0, sizeof(float));  // Important: initialize sum to 0
	cudaMemcpy(Md, M, sizeof(float) * N, cudaMemcpyHostToDevice);

  int threadsPerBlock = THREADDIM;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	// Launch kernels
	ExponentiateKernel<<<blocksPerGrid, threadsPerBlock>>>(Md, N);
    SumReduction<<<blocksPerGrid, threadsPerBlock>>>(Md, sumd, N);


	float sum;
	cudaMemcpy(&sum, sumd, sizeof(float), cudaMemcpyDeviceToHost);

  softmax<<<blocksPerGrid, threadsPerBlock>>>(Md, sumd, N);

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
