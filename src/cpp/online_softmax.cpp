#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

vector<float> online_softmax(vector<float> A, int N){
	float m = A[0];
	float prevm = m;
	float d = 0;

	vector<float> softmax_vector;

	for (int i=1; i < N; i++) {
		float prevm = m;
		m = fmax(prevm, A[i-1]);
		d = d * exp(prevm - m) + exp(A[i] - m);
	}

	for (int i=0; i < N; i++) {
		float expr = exp(A[i] - m);
		softmax_vector.push_back(expr / d);
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
	
	vector<float> out = online_softmax(A, A.size());

	cout << "Done with softmax" << endl;
	print_vector(out, out.size());

	return 0;
}
