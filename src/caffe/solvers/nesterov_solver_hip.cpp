#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void NesterovUpdate(int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate) {
  HIP_KERNEL_LOOP(i, N) {
    float hi = h[i];
    float hi_new = h[i] = momentum * hi + local_rate * g[i];
    g[i] = (1+momentum) * hi_new - momentum * hi;
  }
}
template <typename Dtype>
void nesterov_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate) {
  hipLaunchKernelGGL(NesterovUpdate<Dtype>, // NOLINT_NEXT_LINE(whitespace/operators)
      dim3(CAFFE_GET_BLOCKS(N)), dim3(CAFFE_HIP_NUM_THREADS), 0, 0,
      N, g, h, momentum, local_rate);
}
template void nesterov_update_gpu<float>(int, float*, float*, float, float);
template void nesterov_update_gpu<double>(int, double*, double*, double,
    double);

}  // namespace caffe
