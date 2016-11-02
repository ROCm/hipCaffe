#include <vector>

#include "caffe/layers/threshold_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ThresholdForward(hipLaunchParm lp, const int n, const Dtype threshold,
    const Dtype* in, Dtype* out) {
  HIP_KERNEL_LOOP(index, n) {
    out[index] = in[index] > threshold ? 1 : 0;
  }
}

template <typename Dtype>
void ThresholdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernel(ThresholdForward<Dtype>, dim3(CAFFE_GET_BLOCKS(count)), dim3(CAFFE_HIP_NUM_THREADS), 0, 0,
      count, threshold_, bottom_data, top_data);
}


INSTANTIATE_LAYER_GPU_FORWARD(ThresholdLayer);


}  // namespace caffe
