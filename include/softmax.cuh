#define SAFE_SOFTMAX_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

void launchSoftmaxKernel(const float* A, int n);

#ifdef __cplusplus
}
#endif

#endif
