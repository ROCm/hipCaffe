#include "hip/hip_runtime.h"
#ifdef USE_ACCMI
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
__global__ void sync_conv_groups() { }
#endif

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
#ifdef USE_MIOPEN

#if 0
  LOG(INFO) << "CuDNNConvolutionLayer<Dtype>::Forward_gpu()\n";
#endif
#if 0
  // Fall back to standard Caffe
  ConvolutionLayer<Dtype>::Forward_gpu(bottom, top);
#else
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Foward through MIOpen in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      MIOPEN_CHECK(mlopenConvolutionForward(
          handle_[g],                        // handle
          miopen::dataType<Dtype>::one,      // *alpha
          bottom_descs_[i],                  // xDesc
          bottom_data + bottom_offset_ * g,  // *x
          filter_desc_,                      // wDesc
          weight + this->weight_offset_ * g, // *w
          conv_descs_[i],                    // convDesc
          fwd_algo_[i],                      // algo
          miopen::dataType<Dtype>::zero,     // *beta
          top_descs_[i],                     // yDesc
          top_data + top_offset_ * g,        // *y
          workspace[g],                      // *workspace
          workspace_fwd_sizes_[i]            // workSpaceSize
      ));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + top_offset_ * g,
                               bias_data + bias_offset_ * g);
      }
    }

    // Synchronize the work across groups.
    hipDeviceSynchronize();
  }
#endif
#endif

#ifdef USE_CUDNN
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    //hipLaunchKernel(HIP_KERNEL_NAME(sync_conv_groups), dim3(1), dim3(1), 0, 0, );
    sync_conv_groups<<<1, 1>>>();
  }
#endif
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

#ifdef USE_MIOPEN

#ifndef MIOPEN_BACKWARD
  // TBD
  // Fall back to standard Caffe
  ConvolutionLayer<Dtype>::Backward_gpu(top, propagate_down, bottom);
#else
#if 1
  LOG(INFO) << "CuDNNConvolutionLayer<Dtype>::Backward_gpu()\n";
#endif
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through MIOpen in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        this->backward_gpu_bias(bias_diff + bias_offset_ * g,
                                top_diff + top_offset_ * g);
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[1]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        MIOPEN_CHECK(mlopenConvolutionBackwardWeights(
            handle_[1 * this->group_ + g],          // handle
            miopen::dataType<Dtype>::one,           // *alpha
            top_descs_[i],                          // dyDesc
            top_diff + top_offset_ * g,             // *dy
            bottom_descs_[i],                       // xDesc
            bottom_data + bottom_offset_ * g,       // *x
            conv_descs_[i],                         // convDesc
            bwd_weight_algo_[i],                    // algo
            miopen::dataType<Dtype>::one,           // *beta
            filter_desc_,                           // dwDesc
            weight_diff + this->weight_offset_ * g, // *dw
            workspace[1 * this->group_ + g],        // *workSpace
            workspace_bwd_filter_sizes_[i]          // workSpaceSize
        ));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        MIOPEN_CHECK(mlopenConvolutionBackwardData(
            handle_[2 * this->group_ + g],     // handle
            miopen::dataType<Dtype>::one,      // *alpha
            top_descs_[i],                     // dyDesc
            top_diff + top_offset_ * g,        // *dy
            filter_desc_,                      // wDesc
            weight + this->weight_offset_ * g, // *w
            conv_descs_[i],                    // convDesc
            bwd_data_algo_[i],                 // algo
            miopen::dataType<Dtype>::zero,     // *beta
            bottom_descs_[i],                 // dxDesc
            bottom_diff + bottom_offset_ * g, // *dx
            workspace[2 * this->group_ + g],  // workSpace
            workspace_bwd_data_sizes_[i]      // workSpaceSize
        ));
      }
    }

    // Synchronize the work across groups.
    hipDeviceSynchronize();
  }
#endif
#endif

#if USE_CUDNN
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspace[1*this->group_ + g],
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[2*this->group_ + g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    //hipLaunchKernel(HIP_KERNEL_NAME(sync_conv_groups), dim3(1), dim3(1), 0, 0, );
    sync_conv_groups<<<1, 1>>>();
  }
#endif
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
