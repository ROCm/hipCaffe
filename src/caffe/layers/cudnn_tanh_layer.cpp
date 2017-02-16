#ifdef USE_ACCMI
#include <vector>

#include "caffe/layers/cudnn_tanh_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNTanHLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  TanHLayer<Dtype>::LayerSetUp(bottom, top);
#ifdef USE_MIOPEN
  // initialize MIOpen
  hipStream_t stream = nullptr;
  MIOPEN_CHECK(mlopenCreateWithStream(&handle_, 1, &stream));
  miopen::createTensor4dDesc<Dtype>(&bottom_desc_);
  miopen::createTensor4dDesc<Dtype>(&top_desc_);
  miopen::createActivationDescriptor<Dtype>(&activ_desc_, mlopenActivationTANH);
#endif

#ifdef USE_CUDNN
  // initialize cuDNN
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createActivationDescriptor<Dtype>(&activ_desc_, CUDNN_ACTIVATION_TANH);
#endif

  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNTanHLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  TanHLayer<Dtype>::Reshape(bottom, top);
  const int N = bottom[0]->num();
  const int K = bottom[0]->channels();
  const int H = bottom[0]->height();
  const int W = bottom[0]->width();
#ifdef USE_MIOPEN
  miopen::setTensor4dDesc<Dtype>(&bottom_desc_, N, K, H, W);
  miopen::setTensor4dDesc<Dtype>(&top_desc_, N, K, H, W);
#endif

#ifdef USE_CUDNN
  cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, N, K, H, W);
  cudnn::setTensor4dDesc<Dtype>(&top_desc_, N, K, H, W);
#endif
}

template <typename Dtype>
CuDNNTanHLayer<Dtype>::~CuDNNTanHLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

#ifdef USE_MIOPEN
  mlopenDestroyTensorDescriptor(this->bottom_desc_);
  mlopenDestroyTensorDescriptor(this->top_desc_);
  mlopenDestroy(this->handle_);
#endif

#ifdef USE_CUDNN
  cudnnDestroyTensorDescriptor(this->bottom_desc_);
  cudnnDestroyTensorDescriptor(this->top_desc_);
  cudnnDestroy(this->handle_);
#endif
}

INSTANTIATE_CLASS(CuDNNTanHLayer);

}  // namespace caffe
#endif
