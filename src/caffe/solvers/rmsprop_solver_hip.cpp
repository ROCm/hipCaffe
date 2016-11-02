#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void RMSPropUpdate(hipLaunchParm lp, int N, Dtype* g, Dtype* h,
    Dtype rms_decay, Dtype delta, Dtype local_rate) {
  HIP_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = h[i] = rms_decay*h[i] + (1-rms_decay)*gi*gi;
    g[i] = local_rate * g[i] / (sqrt(hi) + delta);
  }
}
template <typename Dtype>
void rmsprop_update_gpu(int N, Dtype* g, Dtype* h, Dtype rms_decay,
    Dtype delta, Dtype local_rate) {
  hipLaunchKernel(RMSPropUpdate<Dtype>,   // NOLINT_NEXT_LINE(whitespace/operators)
      dim3(CAFFE_GET_BLOCKS(N)), dim3(CAFFE_HIP_NUM_THREADS), 0, 0,
      N, g, h, rms_decay, delta, local_rate);
}
template void rmsprop_update_gpu<float>(int, float*, float*, float, float,
    float);
template void rmsprop_update_gpu<double>(int, double*, double*, double, double,
    double);

}  // namespace caffe
