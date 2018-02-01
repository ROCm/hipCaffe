#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void SGDUpdate(int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate) {
#ifndef NULLIFY_KERNELS
  HIP_KERNEL_LOOP(i, N) {
    g[i] = h[i] = momentum*h[i] + local_rate*g[i];
  }
#endif
}
template <typename Dtype>
void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate) {
  hipLaunchKernelGGL(SGDUpdate<Dtype>,  // NOLINT_NEXT_LINE(whitespace/operators)
      dim3(CAFFE_GET_BLOCKS(N)), dim3(CAFFE_HIP_NUM_THREADS), 0, 0,
      N, g, h, momentum, local_rate);
}
template void sgd_update_gpu<float>(int, float*, float*, float, float);
template void sgd_update_gpu<double>(int, double*, double*, double, double);

}  // namespace caffe
