#ifdef USE_ACCMI
#include <vector>

#include "caffe/layers/cudnn_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::LayerSetUp(bottom, top);

#ifdef USE_MIOPEN
#ifdef USE_MIOPEN_DEVELOP
  hipStream_t stream = nullptr;
  MIOPEN_CHECK(miopenCreateWithStream(&handle_, stream));
#else
  MIOPEN_CHECK(miopenCreate(&handle_));
#endif
  miopen::createTensor4dDesc<Dtype>(&bottom_desc_);
  miopen::createTensor4dDesc<Dtype>(&top_desc_);
  miopen::createPoolingDesc<Dtype>(&pooling_desc_,
      this->layer_param_.pooling_param().pool(), &mode_,
      this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_,
      this->stride_h_, this->stride_w_);
#endif

#ifdef USE_CUDNN
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createPoolingDesc<Dtype>(&pooling_desc_,
      this->layer_param_.pooling_param().pool(), &mode_,
      this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_,
      this->stride_h_, this->stride_w_);
#endif
  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::Reshape(bottom, top);

#ifdef USE_MIOPEN
  miopen::setTensor4dDesc<Dtype>(&bottom_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  miopen::setTensor4dDesc<Dtype>(&top_desc_, bottom[0]->num(),
      this->channels_, this->pooled_height_, this->pooled_width_);

  size_t totalSizeInBytes = 0;
  miopenPoolingGetWorkSpaceSize(top_desc_, &totalSizeInBytes);

  if (totalSizeInBytes > workspaceSize) {
    if (workspaceSize == 0) {
      DLOG(INFO) << "Allocating workspace storage " << this->layer_param().name() << "  " << totalSizeInBytes/1024.0/1024.0 << " MB\n";
    } else {
      DLOG(INFO) << "Reallocating workspace storage " << this->layer_param().name() << "  " << totalSizeInBytes/1024.0/1024.0 << " MB\n";
    }

    workspaceSize = totalSizeInBytes;

    hipFree(workspace);

    HIP_CHECK(hipMalloc(&workspace, workspaceSize));
  }
#endif

#ifdef USE_CUDNN
  cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  cudnn::setTensor4dDesc<Dtype>(&top_desc_, bottom[0]->num(),
      this->channels_, this->pooled_height_, this->pooled_width_);
#endif
}

template <typename Dtype>
CuDNNPoolingLayer<Dtype>::~CuDNNPoolingLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

#ifdef USE_MIOPEN
  miopenDestroyTensorDescriptor(bottom_desc_);
  miopenDestroyTensorDescriptor(top_desc_);
  miopenDestroyPoolingDescriptor(pooling_desc_);
  miopenDestroy(handle_);

  hipFree(workspace);
#endif

#ifdef USE_CUDNN
  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyPoolingDescriptor(pooling_desc_);
  cudnnDestroy(handle_);
#endif
}

INSTANTIATE_CLASS(CuDNNPoolingLayer);

}   // namespace caffe
#endif
