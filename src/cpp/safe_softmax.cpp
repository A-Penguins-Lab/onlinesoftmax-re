#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

float sum(vector<float> A, int N) {
	float sum = 0;

	for (int i=0;i<N;i++){
		sum = sum + A[i];
	}

	return sum;
}

float vector_max(vector<float> A, int N) {
	float max = a[0];
	if (N == 0) {
		return 0.0f;
	} else if (N == 1) {
		return A[0];
	}

	for (int i = 0; i < N; i++) {
		if (A[i] > max) {
			max = A[i];
		}
	}

	return max; 
}

// softmax(v_i) = exp(v_i) / sum(e4xp(v))
vector<float> softmax(vector<float> A, int N) {
	vector<float> exp_vector;
	vector<float> softmax_vector;
	
	// n is the normalization factor
	float n = vector_max(A, N);

	for (int i=0; i<N; i++){
		float expr = exp(A[i] - n);
		exp_vector.push_back(expr);
	}

	float exp_sum = sum(exp_vector, N);
	for(int i =0;i<N;i++){
		softmax_vector.push_back(exp_vector[i]/exp_sum);
	}

	return softmax_vector;
}

void print_vector(vector<float> A, int N) {
	for(int i = 0; i < N; i++) {
		cout << i << A[i] << " ";
	}
}

int main() {
	vector<float> A = {1.4,2.34,5.32,10.8,8.1};

	print_vector(A, A.size());
	
	vector<float> out = softmax(A, A.size());

	cout << "Done with softmax" << endl;
	print_vector(out, out.size());

	return 0;
}
