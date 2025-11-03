#ifdef HOST_FUNCTIONS_H
#define HOST_FUNCTIONS_H

void Softmax(const float* A, int n);

void SafeSoftmax(const float* A, int n);

void OnlineSoftmax(const float* A, int n);

#endif
