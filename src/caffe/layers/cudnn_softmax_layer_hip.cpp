#ifdef USE_ACCMI
#include <vector>

#ifdef USE_CUDNN
#include "thrust/device_vector.h"
#endif

#include "caffe/layers/cudnn_softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

#ifdef USE_MIOPEN
  MIOPEN_CHECK(mlopenSoftmaxForward(
      handle_,                       // handle
      miopen::dataType<Dtype>::one,  // *alpha
      bottom_desc_,                  // xDesc
      bottom_data,                   // *x
      miopen::dataType<Dtype>::zero, // *beta
      top_desc_,                     // yDesc
      top_data                       // *y
  ));
#endif

#ifdef USE_CUDNN
  CUDNN_CHECK(cudnnSoftmaxForward(handle_, CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        top_desc_, top_data));
#endif
}

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

#ifdef USE_MIOPEN
    MIOPEN_CHECK(mlopenSoftmaxBackward(
        handle_,                       // handle
        miopen::dataType<Dtype>::one,  // *alpha
        top_desc_,                     // yDesc
        top_data,                      // *y
        top_desc_,                     // dyDesc
        top_diff,                      // *dy
        miopen::dataType<Dtype>::zero, // *beta
        bottom_desc_,                  // dxDesc
        bottom_diff                    // *dx
    ));
#endif

#ifdef USE_CUDNN
    CUDNN_CHECK(cudnnSoftmaxBackward(handle_, CUDNN_SOFTMAX_ACCURATE,
          CUDNN_SOFTMAX_MODE_CHANNEL,
          cudnn::dataType<Dtype>::one,
          top_desc_, top_data, top_desc_, top_diff,
          cudnn::dataType<Dtype>::zero,
          bottom_desc_, bottom_diff));
#endif
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNSoftmaxLayer);

}  // namespace caffe
#endif
