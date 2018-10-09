#ifdef USE_ACCMI
#include <algorithm>
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {

#define WORKSPACE_PER_GROUP 3

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::CuDNNConvolutionLayer(const LayerParameter& param)
    : ConvolutionLayer<Dtype>(param), handles_setup_(false),
      fwd_algo_(), bwd_weight_algo_(), bwd_data_algo_(),
      workspace_fwd_sizes_(), workspace_bwd_filter_sizes_(), workspace_bwd_data_sizes_(),
      workspace(), handle_(nullptr) { }

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
#ifdef USE_MIOPEN
  assert((this->group_ == 1) && "Error: groups > 1 are not fully tested, and therefore disabled.");

  // Initialize algorithm arrays
  fwd_algo_.resize(bottom.size());
  bwd_weight_algo_.resize(bottom.size());
  bwd_data_algo_.resize(bottom.size());
#endif

  // initialize size arrays
  workspace_fwd_sizes_.resize(bottom.size());
  workspace_bwd_filter_sizes_.resize(bottom.size());
  workspace_bwd_data_sizes_.resize(bottom.size());

  // workspace data
  workspaceSizeInBytes = 0;
  workspaceData = NULL;
  workspace.resize(this->group_ * WORKSPACE_PER_GROUP);

#ifdef USE_MIOPEN
  for (size_t i = 0; i < bottom.size(); ++i) {
    // initialize all to default algorithms
    fwd_algo_[i] = miopenConvolutionFwdAlgoDirect;
    bwd_weight_algo_[i] = miopenConvolutionBwdWeightsAlgoDirect;
    bwd_data_algo_[i] = miopenConvolutionBwdDataAlgoDirect;

    // default algorithms don't require workspace
    workspace_fwd_sizes_[i] = 0;
    workspace_bwd_data_sizes_[i] = 0;
    workspace_bwd_filter_sizes_[i] = 0;
  }
#endif

  for (int g = 0; g < this->group_ * WORKSPACE_PER_GROUP; g++) {
#ifdef USE_MIOPEN
    int device;
    HIP_CHECK(hipGetDevice(&device));

    auto& hmap = caffe::miopen::miopenHandleMap::getInstance();
    handle_ = hmap.getHandle(device);
    if (handle_ == nullptr) {
      DLOG(INFO) << "Creating MIOpen handle on device: " << device;
      MIOPEN_CHECK(miopenCreateWithStream(&handle_, nullptr));
      hmap.setHandle(device, handle_);
    } else {
      DLOG(INFO) << "Get MIOpen handle from cache on device: " << device;
    }
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

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
#ifdef USE_MIOPEN
    miopenTensorDescriptor_t bottom_desc;
    miopen::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    miopenTensorDescriptor_t top_desc;
    miopen::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    miopenConvolutionDescriptor_t conv_desc;
    miopen::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
#endif
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
#ifdef USE_MIOPEN
    miopen::createTensor4dDesc<Dtype>(&bias_desc_);
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

#ifdef USE_MIOPEN
  bool doReshape = false;
  if ((N_ != this->num_) ||
      (C_ != this->channels_ / this->group_) ||
      (H_ != height) ||
      (W_ != width)) {
    doReshape = true;
    DLOG(INFO) << "doReshape\n";
    N_ = this->num_;
    C_ = this->channels_ / this->group_;
    H_ = height;
    W_ = width;
  } else {
    DLOG(INFO) << "NOT doReshape\n";
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


    DLOG(INFO) << "Before miopenConvolution*GetWorkSpaceSize\n";
#ifdef USE_MIOPEN_FORWARD_CONV
    MIOPEN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(
        handle_,                        // handle
        filter_desc_,                   // wDesc
        bottom_descs_[i],               // xDesc
        conv_descs_[i],                 // convDesc
        top_descs_[i],                  // yDesc
        &workspace_fwd_sizes_[i]        // workSpaceSize
    ));

#endif // USE_MIOPEN_FORWARD_CONV

#ifdef USE_MIOPEN_BACKWARD_WEIGHT
    // get workspace for backwards filter algorithm
    MIOPEN_CHECK(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
        handle_,                        // handle
        top_descs_[i],                  // dyDesc
        bottom_descs_[i],               // xDesc
        conv_descs_[i],                 // convDesc
        filter_desc_,                   // dwDesc
        &workspace_bwd_filter_sizes_[i] // workSpaceSize
    ));
#endif // USE_MIOPEN_BACKWARD_WEIGHT

#ifdef USE_MIOPEN_BACKWARD_DATA
    // get workspace for backwards filter algorithm
    MIOPEN_CHECK(miopenConvolutionBackwardDataGetWorkSpaceSize(
        handle_,                        // handle
        top_descs_[i],                  // dyDesc
        filter_desc_,                   // wDesc
        conv_descs_[i],                 // convDesc
        bottom_descs_[i],               // dxDesc
        &workspace_bwd_data_sizes_[i] // workSpaceSize
    ));

    DLOG(INFO) << "After miopenConvolution*GetWorkSpaceSize\n";
#endif // USE_MIOPEN_BACKWARD_DATA
#endif // USE_MIOPEN
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

  // get max over all operations
  size_t max_workspace = std::max(total_workspace_fwd,
                             total_workspace_bwd_data);
  max_workspace = std::max(max_workspace, total_workspace_bwd_filter);

  // ensure all groups have enough workspace
  size_t total_max_workspace = max_workspace *
                               (this->group_ * WORKSPACE_PER_GROUP);


  DLOG(INFO) << "  total_workspace_fwd: " << total_workspace_fwd << "\n";
  DLOG(INFO) << "  total_workspace_bwd_data: " << total_workspace_bwd_data << "\n";
  DLOG(INFO) << "  total_workspace_bwd_filter: " << total_workspace_bwd_filter << "\n";
  DLOG(INFO) << "  max_workspace: " << max_workspace << "\n";
  DLOG(INFO) << "  total_max_workspace: " << total_max_workspace << "\n";

  // this is the total amount of storage needed over all groups + streams
  if (total_max_workspace > workspaceSizeInBytes) {
    if (workspaceSizeInBytes == 0) {
      DLOG(INFO) << "Allocating workspace storage " << this->layer_param().name() << "  " << total_max_workspace/1024.0/1024.0 << " MB\n";
    } else {
      DLOG(INFO) << "Reallocating workspace storage " << this->layer_param().name() << "  " << total_max_workspace/1024.0/1024.0 << " MB\n";
    }

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
        fwd_algo_[i] = miopenConvolutionFwdAlgoDirect;
#ifdef USE_MIOPEN_BACKWARD_CONV
        assert(0); // TODO - Don't have any backward algs that work without workspace memory
        bwd_weight_algo_[i] = miopenConvolutionBwdWeightsAlgoDirect;
        bwd_data_algo_[i] = miopenConvolutionBwdDataAlgoDirect;
#endif
#endif
      }

      // NULL out all workspace pointers
      for (int g = 0; g < (this->group_ * WORKSPACE_PER_GROUP); g++) {
        workspace[g] = NULL;
      }
      // NULL out underlying data
      workspaceData = NULL;
      workspaceSizeInBytes = 0;
    }

    // if we succeed in the allocation, set pointer aliases for workspaces
    for (int g = 0; g < (this->group_ * WORKSPACE_PER_GROUP); g++) {
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

    int ret_algo_count = 0;
    miopenConvAlgoPerf_t perf;
    perf.fwd_algo = miopenConvolutionFwdAlgoDirect;
    perf.bwd_weights_algo = miopenConvolutionBwdWeightsAlgoDirect;
    perf.bwd_data_algo = miopenConvolutionBwdDataAlgoDirect;

#ifdef USE_MIOPEN_FORWARD_CONV
    DLOG(INFO) << "miopenFindConvolutionForwardAlgorithm\n";

    // choose forward and backward algorithms + workspace(s)
    MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
        handle_,                  // handle
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
        workspace[0*this->group_ + g],             // workSpace
        total_workspace_fwd,            // workSpaceSize
        false                     // exhaustiveSearch
    ));

    fwd_algo_[i] = perf.fwd_algo;
    DLOG(INFO) << " - fwd_algo_[" << i << "]:        " << fwd_algo_[i];
    DLOG(INFO) << "workspace_fwd_sizes_[" << i << "]:" << workspace_fwd_sizes_[i] << "\n";
#endif

    const Dtype* top_diff = top[i]->gpu_diff();
#ifdef USE_MIOPEN_BACKWARD_WEIGHT
    DLOG(INFO) << "miopenFindConvolutionBackwardWeightsAlgorithm\n";

    // choose backward algorithm for filter
    MIOPEN_CHECK(miopenFindConvolutionBackwardWeightsAlgorithm(
        handle_,                  // handle
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
        workspace[1*this->group_ + g],             // workSpace
        total_workspace_bwd_filter,            // workSpaceSize
        false                     // exhaustiveSearch
    ));

    bwd_weight_algo_[i] = perf.bwd_weights_algo;
    DLOG(INFO) << " - bwd_weight_algo_[" << i << "]: " << bwd_weight_algo_[i];
    DLOG(INFO) << "workspace_bwd_filter_sizes_[" << i << "]: " << workspace_bwd_filter_sizes_[i] << "\n";
#endif

    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
#ifdef USE_MIOPEN_BACKWARD_DATA
    if (this->param_propagate_down_[i]) {
      DLOG(INFO) << "miopenFindConvolutionBackwardDataAlgorithm\n";
  
      // choose backward algo for data
      MIOPEN_CHECK(miopenFindConvolutionBackwardDataAlgorithm(
          handle_,                  // handle
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
          workspace[2*this->group_ + g],             // workSpace
          total_workspace_bwd_data,            // workSpaceSize
          false                     // exhaustiveSearch
      ));
  
      bwd_data_algo_[i] = perf.bwd_data_algo;
      DLOG(INFO) << " - bwd_data_algo_[" << i << "]:   " << bwd_data_algo_[i];
      DLOG(INFO) << "workspace_bwd_data_sizes_[" << i << "]: " << workspace_bwd_data_sizes_[i] << "\n";
    }
#endif // USE_MIOPEN_BACKWARD_DATA

   } // For g
  } //for i
  } // doReshape
#endif // USE_MIOPEN

  // Tensor descriptor for bias.
  if (this->bias_term_) {
#ifdef USE_MIOPEN
    miopen::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
#endif
  }
}

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { 
    return;
  }

#ifdef USE_MIOPEN
  for (int i = 0; i < bottom_descs_.size(); i++) {
    miopenDestroyTensorDescriptor(bottom_descs_[i]);
    miopenDestroyTensorDescriptor(top_descs_[i]);
    miopenDestroyConvolutionDescriptor(conv_descs_[i]);
  }
#endif

  if (this->bias_term_) {
#ifdef USE_MIOPEN
    miopenDestroyTensorDescriptor(bias_desc_);
#endif
  }
#ifdef USE_MIOPEN
#ifdef USE_MIOPEN_FORWARD_CONV
  miopenDestroyTensorDescriptor(filter_desc_);
#endif
#endif

  hipFree(workspaceData);
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}   // namespace caffe
#endif
