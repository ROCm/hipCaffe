#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void AdaGradUpdate(int N, Dtype* g, Dtype* h, Dtype delta,
    Dtype local_rate) {
#ifndef NULLIFY_KERNELS
  HIP_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = h[i] = h[i] + gi*gi;
    g[i] = local_rate * gi / (sqrt(hi) + delta);
  }
#endif
}
template <typename Dtype>
void adagrad_update_gpu(int N, Dtype* g, Dtype* h, Dtype delta,
    Dtype local_rate) {
  hipLaunchKernelGGL(AdaGradUpdate<Dtype>,  // NOLINT_NEXT_LINE(whitespace/operators)
      dim3(CAFFE_GET_BLOCKS(N)), dim3(CAFFE_HIP_NUM_THREADS), 0, 0,
      N, g, h, delta, local_rate);
}
template void adagrad_update_gpu<float>(int, float*, float*, float, float);
template void adagrad_update_gpu<double>(int, double*, double*, double, double);

}  // namespace caffe
