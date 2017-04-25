#include <algorithm>
#include <vector>

#include "caffe/layers/bnll_layer.hpp"

namespace caffe {

const float kBNLL_THRESHOLD = 50.;

template <typename Dtype>
__global__ void BNLLForward(const int n, const Dtype* in, Dtype* out) {
  HIP_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ?
        in[index] + log(1. + exp(-in[index])) :
        log(1. + exp(in[index]));
  }
}

template <typename Dtype>
void BNLLLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernel(BNLLForward<Dtype>, dim3(CAFFE_GET_BLOCKS(count)), dim3(CAFFE_HIP_NUM_THREADS), 0, 0,
      count, bottom_data, top_data);
  //HIP_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void BNLLBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  HIP_KERNEL_LOOP(index, n) {
    Dtype expval = exp(min(in_data[index], Dtype(kBNLL_THRESHOLD)));
    out_diff[index] = in_diff[index] * expval / (expval + 1.);
  }
}

template <typename Dtype>
void BNLLLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    hipLaunchKernel(BNLLBackward<Dtype>, dim3(CAFFE_GET_BLOCKS(count)), dim3(CAFFE_HIP_NUM_THREADS), 0, 0,
        count, top_diff, bottom_data, bottom_diff);
    //HIP_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BNLLLayer);


}  // namespace caffe
