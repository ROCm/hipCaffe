#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void AdaDeltaUpdate(int N, Dtype* g, Dtype* h, Dtype* h2,
    Dtype momentum, Dtype delta, Dtype local_rate) {
  HIP_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = h[i] = momentum * h[i] + (1-momentum) * gi * gi;
    gi = gi * sqrt((h2[i] + delta) / (hi + delta));
    h2[i] = momentum * h2[i] + (1-momentum) * gi * gi;
    g[i] = local_rate * gi;
  }
}
template <typename Dtype>
void adadelta_update_gpu(int N, Dtype* g, Dtype* h, Dtype* h2, Dtype momentum,
    Dtype delta, Dtype local_rate) {
  hipLaunchKernelGGL(AdaDeltaUpdate<Dtype>,   // NOLINT_NEXT_LINE(whitespace/operators)
      dim3(CAFFE_GET_BLOCKS(N)), dim3(CAFFE_HIP_NUM_THREADS), 0, 0, 
      N, g, h, h2, momentum, delta, local_rate);
}
template void adadelta_update_gpu<float>(int , float*, float*, float*,
    float, float, float);
template void adadelta_update_gpu<double>(int, double*, double*, double*,
    double, double, double);

}  // namespace caffe
