#ifdef USE_ACCMI
#include <algorithm>
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently

#define CUDNN_STREAMS_PER_GROUP 3

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
#ifdef USE_MIOPEN
  // Initalize HIP streams and MIOpen.
  stream_         = new hipStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  handle_         = new mlopenHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];

  // Initialize algorithm arrays
  fwd_algo_       = new mlopenConvFwdAlgorithm_t[bottom.size()];
  bwd_weight_algo_= new mlopenConvBwdWeightsAlgorithm_t[bottom.size()];
  bwd_data_algo_  = new mlopenConvBwdDataAlgorithm_t[bottom.size()];
#endif
#ifdef USE_CUDNN
  // Initialize CUDA streams and cuDNN.
  stream_         = new hipStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  handle_         = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];

  // Initialize algorithm arrays
  fwd_algo_       = new cudnnConvolutionFwdAlgo_t[bottom.size()];
  bwd_filter_algo_= new cudnnConvolutionBwdFilterAlgo_t[bottom.size()];
  bwd_data_algo_  = new cudnnConvolutionBwdDataAlgo_t[bottom.size()];
#endif

  // initialize size arrays
  workspace_fwd_sizes_ = new size_t[bottom.size()];
  workspace_bwd_filter_sizes_ = new size_t[bottom.size()];
  workspace_bwd_data_sizes_ = new size_t[bottom.size()];

  // workspace data
  workspaceSizeInBytes = 0;
  workspaceData = NULL;
  workspace = new void*[this->group_ * CUDNN_STREAMS_PER_GROUP];

#ifdef USE_MIOPEN
  for (size_t i = 0; i < bottom.size(); ++i) {
    // initialize all to default algorithms
    fwd_algo_[i] = mlopenConvolutionFwdAlgoDirect;
    bwd_weight_algo_[i] = mlopenConvolutionBwdWeightsAlgoDirect;
    bwd_data_algo_[i] = mlopenConvolutionBwdDataAlgoDirect;

    // default algorithms don't require workspace
    workspace_fwd_sizes_[i] = 0;
    workspace_bwd_data_sizes_[i] = 0;
    workspace_bwd_filter_sizes_[i] = 0;
  }
#endif

#ifdef USE_CUDNN
  for (size_t i = 0; i < bottom.size(); ++i) {
    // initialize all to default algorithms
    fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
    bwd_filter_algo_[i] = (cudnnConvolutionBwdFilterAlgo_t)0;
    bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t)0;
    // default algorithms don't require workspace
    workspace_fwd_sizes_[i] = 0;
    workspace_bwd_data_sizes_[i] = 0;
    workspace_bwd_filter_sizes_[i] = 0;
  }
#endif

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
#ifdef USE_MIOPEN
    HIP_CHECK(hipStreamCreate(&stream_[g]));
#ifdef USE_MIOPEN_DEVELOP
    MIOPEN_CHECK(mlopenCreateWithStream(&handle_[g], &stream_[g]));
#else
    MIOPEN_CHECK(mlopenCreate(&handle_[g], 1, &stream_[g]));
#endif
#endif

#ifdef USE_CUDNN
    HIP_CHECK(hipStreamCreate(&stream_[g]));
    CUDNN_CHECK(cudnnCreate(&handle_[g]));
    CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
#endif
    workspace[g] = NULL;
  }

  // Set the indexing parameters.
  bias_offset_ = (this->num_output_ / this->group_);

  // Create filter descriptor.
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];
#ifdef USE_MIOPEN_FORWARD_CONV
  miopen::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      kernel_h, kernel_w);
#endif
#ifdef USE_CUDNN
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      kernel_h, kernel_w);
#endif

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
#ifdef USE_MIOPEN
    mlopenTensorDescriptor_t bottom_desc;
    miopen::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    mlopenTensorDescriptor_t top_desc;
    miopen::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    mlopenConvolutionDescriptor_t conv_desc;
    miopen::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
#endif

#ifdef USE_CUDNN
    cudnnTensorDescriptor_t bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    cudnnTensorDescriptor_t top_desc;
    cudnn::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
#endif
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
#ifdef USE_MIOPEN
    miopen::createTensor4dDesc<Dtype>(&bias_desc_);
#endif

#ifdef USE_CUDNN
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
#endif
  }

#ifdef USE_MIOPEN
  N_ = C_ = W_ = H_ = 0;
#endif

  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(2, this->num_spatial_axes_)
      << "CuDNNConvolution input must have 2 spatial axes "
      << "(e.g., height and width). "
      << "Use 'engine: CAFFE' for general ND convolution.";
  bottom_offset_ = this->bottom_dim_ / this->group_;
  top_offset_ = this->top_dim_ / this->group_;
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

#ifdef USE_MIOPEN
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
#endif

#ifdef USE_CUDNN
  // Specify workspace limit for kernels directly until we have a
  // planning strategy and a rewrite of Caffe's GPU memory mangagement
  size_t workspace_limit_bytes = 8*1024*1024;
#endif

#ifdef USE_MIOPEN
  bool doReshape = false;
  if ((N_ != this->num_) ||
      (C_ != this->channels_ / this->group_) ||
      (H_ != height) ||
      (W_ != width)) {
    doReshape = true;
    //LOG(INFO) << "doReshape\n";
    N_ = this->num_;
    C_ = this->channels_ / this->group_;
    H_ = height;
    W_ = width;
  } else {
    //LOG(INFO) << "NOT doReshape\n";
  }

  if (doReshape) {
#endif

  for (int i = 0; i < bottom.size(); i++) {
#ifdef USE_MIOPEN
    miopen::setTensor4dDesc<Dtype>(&bottom_descs_[i],
        this->num_,
        this->channels_ / this->group_, height, width,
        this->channels_ * height * width,
        height * width, width, 1);
    miopen::setTensor4dDesc<Dtype>(&top_descs_[i],
        this->num_,
        this->num_output_ / this->group_, height_out, width_out,
        this->num_output_ * this->out_spatial_dim_,
        this->out_spatial_dim_, width_out, 1);
    miopen::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
        filter_desc_, pad_h, pad_w,
        stride_h, stride_w);



#ifdef USE_MIOPEN_BACKWARD_WEIGHT
    LOG(INFO) << "Before mlopenConvolutionBackwardWeightsGetWorkSpaceSize\n";
    // get workspace for backwards filter algorithm
    MIOPEN_CHECK(mlopenConvolutionBackwardWeightsGetWorkSpaceSize(
        top_descs_[i],                  // dyDesc
        bottom_descs_[i],               // xDesc
        conv_descs_[i],                 // convDesc
        filter_desc_,                   // dwDesc
        &workspace_bwd_filter_sizes_[i] // workSpaceSize
    ));
    LOG(INFO) << "After mlopenConvolutionBackwardWeightsGetWorkSpaceSize\n";
    LOG(INFO) << "workspace_bwd_filter_sizes_[" << i << "]:" << workspace_bwd_filter_sizes_[i] << "\n";
#endif // USE_MIOPEN_BACKWARD_WEIGHT
#endif



#ifdef USE_CUDNN
    cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
        this->num_,
        this->channels_ / this->group_, height, width,
        this->channels_ * height * width,
        height * width, width, 1);
    cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
        this->num_,
        this->num_output_ / this->group_, height_out, width_out,
        this->num_output_ * this->out_spatial_dim_,
        this->out_spatial_dim_, width_out, 1);
    cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
        filter_desc_, pad_h, pad_w,
        stride_h, stride_w);

    // choose forward and backward algorithms + workspace(s)
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
      workspace_limit_bytes,
      &fwd_algo_[i]));

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      fwd_algo_[i],
      &(workspace_fwd_sizes_[i])));

    // choose backward algorithm for filter
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle_[0],
          bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          workspace_limit_bytes, &bwd_filter_algo_[i]) );

    // get workspace for backwards filter algorithm
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_[0],
          bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
          bwd_filter_algo_[i], &workspace_bwd_filter_sizes_[i]));

    // choose backward algo for data
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle_[0],
          filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
          CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes, &bwd_data_algo_[i]));

    // get workspace size
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_[0],
          filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
          bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]) );
#endif
  }

#ifdef USE_MIOPEN
  } // if (doReshape)
#endif

  // reduce over all workspace sizes to get a maximum to allocate / reallocate
  size_t total_workspace_fwd = 0;
  size_t total_workspace_bwd_data = 0;
  size_t total_workspace_bwd_filter = 0;

  for (size_t i = 0; i < bottom.size(); i++) {
    total_workspace_fwd        = std::max(total_workspace_fwd,
                                     workspace_fwd_sizes_[i]);
    total_workspace_bwd_data   = std::max(total_workspace_bwd_data,
                                     workspace_bwd_data_sizes_[i]);
    total_workspace_bwd_filter = std::max(total_workspace_bwd_filter,
                                     workspace_bwd_filter_sizes_[i]);
  }
#if 0
  LOG(INFO) << "total_workspace_fwd: " << total_workspace_fwd << "\n";
  LOG(INFO) << "total_workspace_bwd_data: " << total_workspace_bwd_data << "\n";
  LOG(INFO) << "total_workspace_bwd_filter: " << total_workspace_bwd_filter << "\n";
#endif

  // get max over all operations
  size_t max_workspace = std::max(total_workspace_fwd,
                             total_workspace_bwd_data);
  max_workspace = std::max(max_workspace, total_workspace_bwd_filter);
#if 0
  LOG(INFO) << "max_workspace: " << max_workspace << "\n";
#endif
  // ensure all groups have enough workspace
  size_t total_max_workspace = max_workspace *
                               (this->group_ * CUDNN_STREAMS_PER_GROUP);
#if 0
  LOG(INFO) << "total_max_workspace: " << total_max_workspace << "\n";
#endif

  // this is the total amount of storage needed over all groups + streams
  if (total_max_workspace > workspaceSizeInBytes) {
    DLOG(INFO) << "Reallocating workspace storage " << this->layer_param().name() << "  " << total_max_workspace/1024.0/1024.0 << " MB\n";

    workspaceSizeInBytes = total_max_workspace;

    // free the existing workspace and allocate a new (larger) one
    hipFree(this->workspaceData);

    hipError_t err = hipMalloc(&(this->workspaceData), workspaceSizeInBytes);
    if (err != hipSuccess) {
      // force zero memory path
      for (int i = 0; i < bottom.size(); i++) {
        DLOG(INFO) << "warning: GPU memory exhausted trying to allocate workspace\n";
        workspace_fwd_sizes_[i] = 0;
        workspace_bwd_filter_sizes_[i] = 0;
        workspace_bwd_data_sizes_[i] = 0;
#ifdef USE_MIOPEN
        fwd_algo_[i] = mlopenConvolutionFwdAlgoDirect;
#ifdef USE_MIOPEN_BACKWARD_CONV
        assert(0); // Don't have any backward algs that work without workspace memory
        bwd_weight_algo_[i] = mlopenConvolutionBwdWeightsAlgoDirect;
        bwd_data_algo_[i] = mlopenConvolutionBwdDataAlgoDirect;
#endif
#endif
#ifdef USE_CUDNN
        fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        bwd_filter_algo_[i] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        bwd_data_algo_[i] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
#endif
      }

      // NULL out all workspace pointers
      for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
        workspace[g] = NULL;
      }
      // NULL out underlying data
      workspaceData = NULL;
      workspaceSizeInBytes = 0;
    }

    // if we succeed in the allocation, set pointer aliases for workspaces
    for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
      workspace[g] = reinterpret_cast<char *>(workspaceData) + g*max_workspace;
    }
  }



#ifdef USE_MIOPEN
  // A second loop for USE_MIOPEN, after we have allocated a workspace:
  if (doReshape) {
  for (int i = 0; i < bottom.size(); i++) {
   for (int g = 0; g < this->group_; g++) {
       // Currently MIOpen requires calling Find* even if the kernel has already been found by another call 
       // (ie a different group will find same kernels).
       // In future might be able to optimize this so don't need to re-find the kernel and could
       // thus only call Find on the group=0 handles:

    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    int ret_algo_count;
    mlopenConvAlgoPerf_t perf;

#ifdef USE_MIOPEN_FORWARD_CONV
    // choose forward and backward algorithms + workspace(s)
    MIOPEN_CHECK(mlopenFindConvolutionForwardAlgorithm(
        handle_[0*this->group_ + g],               // handle
        bottom_descs_[i],         // xDesc
        bottom_data,              // *x
        filter_desc_,             // wDesc
        weight,                   // *w
        conv_descs_[i],           // convDesc
        top_descs_[i],            // yDesc
        top_data,                 // *y
        1,                        // requestAlgoCount
        &ret_algo_count,          // returnedAlgoCount
        &perf,                    // perfResults
        NULL,                     // workSpace
        0,                        // workSpaceSize
        false                     // exhaustiveSearch
    ));
#endif

    // TODO - currently overriding MLOpen fwd_algo to work around bug, will re-enable when fwd_algo is valid
    //fwd_algo_[i] = perf.fwd_algo;
    fwd_algo_[i] = mlopenConvolutionFwdAlgoDirect;
#if 0
    workspace_fwd_sizes_[i] = perf.memory; // TODO
#endif
#if 0
    LOG(INFO) << "fwd_algo_[" << i << "]: " << fwd_algo_[i] << "\n";
    LOG(INFO) << "workspace_fwd_sizes_[" << i << "]:" << workspace_fwd_sizes_[i] << "\n";
#endif

    const Dtype* top_diff = top[i]->gpu_diff();
#ifdef USE_MIOPEN_BACKWARD_WEIGHT

    LOG(INFO) << "Before mlopenFindConvolutionBackwardWeightsAlgorithm\n";
    // choose backward algorithm for filter
    MIOPEN_CHECK(mlopenFindConvolutionBackwardWeightsAlgorithm(
        handle_[1*this->group_ + g],               // handle
        top_descs_[i],            // dyDesc
        top_diff,                 // *dy
        bottom_descs_[i],         // xDesc
        bottom_data,              // *x
        conv_descs_[i],           // convDesc
        filter_desc_,             // dwDesc
        weight_diff,              // *dw
        1,                        // requestAlgoCount
        &ret_algo_count,          // returnedAlgoCount
        &perf,                    // perfResults
        workspace[0],             // workSpace
        max_workspace,            // workSpaceSize
        false                     // exhaustiveSearch
    ));
    LOG(INFO) << "After mlopenFindConvolutionBackwardWeightsAlgorithm\n";

    bwd_weight_algo_[i] = perf.bwd_weights_algo;
    LOG(INFO) << "  bwd_weight_algo_[" << i << "]: " << bwd_weight_algo_[i] << "\n";
    //workspace_bwd_filter_sizes_[i] = perf.memory; // TODO-MIOpen perf
#endif

#ifdef USE_MIOPEN_BACKWARD_DATA
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();

    LOG(INFO) << "Before mlopenFindConvolutionBackwardDataAlgorithm\n";
    // choose backward algo for data
    MIOPEN_CHECK(mlopenFindConvolutionBackwardDataAlgorithm(
        handle_[2*this->group_ + g],               // handle
        top_descs_[i],            // dyDesc
        top_diff,                 // *dy
        filter_desc_,             // wDesc
        weight,                   // *w
        conv_descs_[i],           // convDesc
        bottom_descs_[i],         // dxDesc
        bottom_diff,              // *dx
        1,                        // requestAlgoCount
        &ret_algo_count,          // returnedAlgoCount
        &perf,                    // perfResults
        workspace[0],             // workSpace
        max_workspace,            // workSpaceSize
        false                     // exhaustiveSearch
    ));
    LOG(INFO) << "After mlopenFindConvolutionBackwardDataAlgorithm perf.memory=" << perf.memory << "\n";

    //bwd_data_algo_[i] = perf.bwd_data_algo; // TODO
    bwd_data_algo_[i] = mlopenConvolutionBwdDataAlgoDirect;

    //workspace_bwd_data_sizes_[i] = perf.memory; // TODO
#endif // USE_MIOPEN_BACKWARD_DATA
    LOG(INFO) << "bwd_data_algo_[" << i << "]: " << bwd_data_algo_[i] << "\n";
    LOG(INFO) << "workspace_bwd_data_sizes_[" << i << "]: " << workspace_bwd_data_sizes_[i] << "\n";

#endif // USE_MIOPEN
   } // For g
  } //for i
  } // doReshape

  // Tensor descriptor for bias.
  if (this->bias_term_) {
#ifdef USE_MIOPEN
    miopen::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
#endif

#ifdef USE_CUDNN
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
#endif
  }
}

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

#ifdef USE_MIOPEN
  for (int i = 0; i < bottom_descs_.size(); i++) {
    mlopenDestroyTensorDescriptor(bottom_descs_[i]);
    mlopenDestroyTensorDescriptor(top_descs_[i]);
    mlopenDestroyConvolutionDescriptor(conv_descs_[i]);
  }
#endif

#ifdef USE_CUDNN
  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensorDescriptor(bottom_descs_[i]);
    cudnnDestroyTensorDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
#endif
  if (this->bias_term_) {
#ifdef USE_MIOPEN
    mlopenDestroyTensorDescriptor(bias_desc_);
#endif

#ifdef USE_CUDNN
    cudnnDestroyTensorDescriptor(bias_desc_);
#endif
  }
#ifdef USE_MIOPEN
  mlopenDestroyTensorDescriptor(filter_desc_);
#endif
#ifdef USE_CUDNN
  cudnnDestroyFilterDescriptor(filter_desc_);
#endif

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
#ifdef USE_MIOPEN
    mlopenDestroy(handle_[g]);
    hipStreamDestroy(stream_[g]);
#endif
#ifdef USE_CUDNN
    hipStreamDestroy(stream_[g]);
    cudnnDestroy(handle_[g]);
#endif
  }

  hipFree(workspaceData);
#ifdef USE_MIOPEN
  delete [] stream_;
  delete [] handle_;
  delete [] fwd_algo_;
  delete [] bwd_weight_algo_;
  delete [] bwd_data_algo_;
#endif
#ifdef USE_CUDNN
  delete [] stream_;
  delete [] handle_;
  delete [] fwd_algo_;
  delete [] bwd_filter_algo_;
  delete [] bwd_data_algo_;
#endif
  delete [] workspace_fwd_sizes_;
  delete [] workspace_bwd_data_sizes_;
  delete [] workspace_bwd_filter_sizes_;
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}   // namespace caffe
#endif
