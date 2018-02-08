#ifdef USE_ACCMI
#include <vector>

#include "caffe/layers/cudnn_lcn_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNLCNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LRNLayer<Dtype>::LayerSetUp(bottom, top);

#ifdef USE_MIOPEN
#ifdef USE_MIOPEN_DEVELOP
  hipStream_t stream = nullptr;
  MIOPEN_CHECK(miopenCreateWithStream(&handle_, stream));
#else
  MIOPEN_CHECK(miopenCreate(&handle_));
#endif
  MIOPEN_CHECK(miopenCreateLRNDescriptor(&norm_desc_));
  miopen::createTensor4dDesc<Dtype>(&bottom_desc_);
  miopen::createTensor4dDesc<Dtype>(&top_desc_);
#endif

#ifdef USE_CUDNN
  CUDNN_CHECK(cudnnCreate(&handle_));
  CUDNN_CHECK(cudnnCreateLRNDescriptor(&norm_desc_));
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
#endif

  // create a LRN handle
  handles_setup_ = true;

  size_ = this->layer_param().lrn_param().local_size();
  pre_pad_ = (size_ - 1) / 2;
  alpha_ = this->layer_param().lrn_param().alpha();
  beta_ = this->layer_param().lrn_param().beta();
  k_ = this->layer_param().lrn_param().k();
}

template <typename Dtype>
void CuDNNLCNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LRNLayer<Dtype>::Reshape(bottom, top);
#ifdef USE_MIOPEN
  miopen::setTensor4dDesc<Dtype>(&bottom_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  miopen::setTensor4dDesc<Dtype>(&top_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  MIOPEN_CHECK(miopenSetLRNDescriptor(norm_desc_, miopenLRNCrossChannel, size_, alpha_, beta_, k_));
#endif

#ifdef USE_CUDNN
  cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  cudnn::setTensor4dDesc<Dtype>(&top_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  CUDNN_CHECK(cudnnSetLRNDescriptor(norm_desc_, size_, alpha_, beta_, k_));
#endif

  // allocate / reallocate tempData buffers
  size_t totalSizeInBytes = sizeof(Dtype)*bottom[0]->num()* \
                            this->channels_*this->height_*this->width_;

  if (totalSizeInBytes > tempDataSize) {
    // Note: reporting two reallocations because of the two hipMallocs below
    if (tempDataSize == 0) {
      DLOG(INFO) << "Allocating temp storage " << this->layer_param().name() << "  " << totalSizeInBytes/1024.0/1024.0 << " MB\n";
      DLOG(INFO) << "Allocating temp storage " << this->layer_param().name() << "  " << totalSizeInBytes/1024.0/1024.0 << " MB\n";
    } else {
      DLOG(INFO) << "Reallocating temp storage " << this->layer_param().name() << "  " << totalSizeInBytes/1024.0/1024.0 << " MB\n";
      DLOG(INFO) << "Reallocating temp storage " << this->layer_param().name() << "  " << totalSizeInBytes/1024.0/1024.0 << " MB\n";
    }

    tempDataSize = totalSizeInBytes;

    hipFree(tempData1);
    hipFree(tempData2);

    // allocate new buffers
    HIP_CHECK(hipMalloc(&tempData1, totalSizeInBytes));
    HIP_CHECK(hipMalloc(&tempData2, totalSizeInBytes));
  }
}

template <typename Dtype>
CuDNNLCNLayer<Dtype>::~CuDNNLCNLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

#ifdef USE_MIOPEN
  miopenDestroyTensorDescriptor(bottom_desc_);
  miopenDestroyTensorDescriptor(top_desc_);

  // destroy LRN handle
  miopenDestroy(handle_);
#endif

#ifdef USE_CUDNN
  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);

  // destroy LRN handle
  cudnnDestroy(handle_);
#endif

  // free temp buffers
  hipFree(tempData1);
  hipFree(tempData2);
}

INSTANTIATE_CLASS(CuDNNLCNLayer);

}   // namespace caffe
#endif
