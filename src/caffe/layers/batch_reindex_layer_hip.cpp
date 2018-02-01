#include <algorithm>
#include <utility>
#include <vector>

#include "caffe/layers/batch_reindex_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
__global__ void BRForward(const int count, const int inner_dim, const Dtype* in,
                          const Dtype* permut, Dtype* out) {
#ifndef NULLIFY_KERNELS
  HIP_KERNEL_LOOP(index, count) {
    int n = index / (inner_dim);
    int in_n = static_cast<int>(permut[n]);
    out[index] = in[in_n * (inner_dim) + index % (inner_dim)];
  }
#endif
}

template<typename Dtype>
void BatchReindexLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  check_batch_reindex(bottom[0]->shape(0), bottom[1]->count(),
                      bottom[1]->cpu_data());
  if (top[0]->count() == 0) {
    return;
  }
  int threads = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  auto top0_count = top[0]->count();
  auto bot0_count = bottom[0]->count(); 
  auto bot0_shape = bottom[0]->shape(0);
  auto bot0_gpu_data = bottom[0]->gpu_data();
  auto bot1_gpu_data = bottom[1]->gpu_data(); 
  auto top0_mut_gpu_data = top[0]->mutable_gpu_data();
  hipLaunchKernelGGL(BRForward, dim3(CAFFE_GET_BLOCKS(threads)), dim3(CAFFE_HIP_NUM_THREADS), 0, 0, top0_count, bot0_count / bot0_shape, bot0_gpu_data, bot1_gpu_data, top0_mut_gpu_data);
  //HIP_POST_KERNEL_CHECK;
}

template<typename Dtype>
__global__ void BRBackward(const int count, const int inner_dim,
                           const Dtype* in, const Dtype* top_indexes,
                           const Dtype* begins, const Dtype* counts,
                           Dtype* out) {
#ifndef NULLIFY_KERNELS
  HIP_KERNEL_LOOP(index, count) {
    int n = index / (inner_dim);
    out[index] = 0;
    int lower = static_cast<int>(begins[n]);
    int upper = lower + static_cast<int>(counts[n]);
    for (int i = lower; i < upper; ++i) {
      int in_n = static_cast<int>(top_indexes[i]);
      out[index] += in[in_n * (inner_dim) + index % (inner_dim)];
    }
  }
#endif
}

template<typename Dtype>
void BatchReindexLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[1]) << "Cannot backprop to index.";
  if (!propagate_down[0]) {
    return;
  }

  vector<std::pair<int, int> > mapping;
  const Dtype* perm = bottom[1]->cpu_data();
  for (int i = 0; i < bottom[1]->count(); ++i) {
    mapping.push_back(pair<int, int>(static_cast<int>(perm[i]), i));
  }
  std::sort(mapping.begin(), mapping.end(), pair_sort_first());

  // Each element of the bottom diff is potentially the sum of many top diffs.
  // However, we'd like each HIP thread to handle exactly one output.  Hence,
  // we first pre-compute a list of lists of indices that need to be summed for
  // each output. `top_indexes` holds the data of this list of lists.  The
  // k'th element of `begins` points to the location in `top_indexes` where the
  // list for the k'th example begin, and the k'th element of `counts` is the
  // length of that list.
  vector<int> shape;
  shape.push_back(bottom[1]->count());
  Blob<Dtype> top_indexes(shape);
  shape[0] = bottom[0]->shape(0);
  Blob<Dtype> counts(shape);
  Blob<Dtype> begins(shape);
  Dtype* t_i_data = top_indexes.mutable_cpu_data();
  Dtype* c_data = counts.mutable_cpu_data();
  Dtype* b_data = begins.mutable_cpu_data();
  caffe_set(begins.count(), Dtype(-1), b_data);
  caffe_set(counts.count(), Dtype(0), c_data);
  for (int i = 0; i < mapping.size(); ++i) {
    t_i_data[i] = mapping[i].second;
    if (b_data[mapping[i].first] == -1) {
      b_data[mapping[i].first] = i;
    }
    c_data[mapping[i].first] += 1;
  }

  int threads = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  auto bot0_count = bottom[0]->count();
  auto bot0_shape = bottom[0]->shape(0);
  auto top0_gpu_diff = top[0]->gpu_diff();
  auto top_idx_gpu_data = top_indexes.gpu_data();
  auto begins_gpu_data= begins.gpu_data();
  auto counts_gpu_data = counts.gpu_data();
  auto bot0_mut_gpu_diff = bottom[0]->mutable_gpu_diff();
  hipLaunchKernelGGL(BRBackward<Dtype>, dim3(CAFFE_GET_BLOCKS(threads)), dim3(CAFFE_HIP_NUM_THREADS), 0, 0, 
      bot0_count, bot0_count / bot0_shape,
      top0_gpu_diff, top_idx_gpu_data, begins_gpu_data,
      counts_gpu_data, bot0_mut_gpu_diff);
  //HIP_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchReindexLayer);

}  // namespace caffe
