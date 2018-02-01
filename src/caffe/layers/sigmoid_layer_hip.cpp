#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SigmoidForward(const int n, const Dtype* in, Dtype* out) {
#ifndef NULLIFY_KERNELS
  HIP_KERNEL_LOOP(index, n) {
    out[index] = 1. / (1. + exp(-in[index]));
  }
#endif
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(SigmoidForward<Dtype>, dim3(CAFFE_GET_BLOCKS(count)), dim3(CAFFE_HIP_NUM_THREADS), 0, 0,
      count, bottom_data, top_data);
  //HIP_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_HIP_NUM_THREADS;
}

template <typename Dtype>
__global__ void SigmoidBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff) {
#ifndef NULLIFY_KERNELS
  HIP_KERNEL_LOOP(index, n) {
    const Dtype sigmoid_x = out_data[index];
    out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);
  }
#endif
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    hipLaunchKernelGGL(SigmoidBackward<Dtype>, dim3(CAFFE_GET_BLOCKS(count)), dim3(CAFFE_HIP_NUM_THREADS), 0, 0,
        count, top_diff, top_data, bottom_diff);
    //HIP_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidLayer);


}  // namespace caffe
