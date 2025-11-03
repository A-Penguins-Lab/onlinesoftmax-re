#define SAFE_SOFTMAX_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

void launchSafeSoftmaxKernel(const float* A, int n);

#ifdef __cplusplus
}
#endif

#endif
