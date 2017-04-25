// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/tanh_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void TanHForward(const int n, const Dtype* in, Dtype* out) {
  HIP_KERNEL_LOOP(index, n) {
    out[index] = tanh(in[index]);
  }
}

template <typename Dtype>
void TanHLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernel(TanHForward<Dtype>, dim3(CAFFE_GET_BLOCKS(count)), dim3(CAFFE_HIP_NUM_THREADS), 0, 0, 
      count, bottom_data, top_data);
  //HIP_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void TanHBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff) {
  HIP_KERNEL_LOOP(index, n) {
    Dtype tanhx = out_data[index];
    out_diff[index] = in_diff[index] * (1 - tanhx * tanhx);
  }
}

template <typename Dtype>
void TanHLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
   hipLaunchKernel(TanHBackward<Dtype>, dim3(CAFFE_GET_BLOCKS(count)), dim3(CAFFE_HIP_NUM_THREADS), 0, 0, 
        count, top_diff, top_data, bottom_diff);
    //HIP_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TanHLayer);


}  // namespace caffe
