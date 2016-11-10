#ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
#define CAFFE_UTIL_DEVICE_ALTERNATE_H_

#ifdef CPU_ONLY  // CPU-only Caffe.

#include <vector>

// Stub out GPU calls as unavailable.

#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."

#define STUB_GPU(classname) \
template <typename Dtype> \
void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \
template <typename Dtype> \
void classname<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

#define STUB_GPU_FORWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \

#define STUB_GPU_BACKWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

#else  // Normal GPU + CPU Caffe.

#include <hip/hip_runtime.h>
#include <hipblas.h>
#ifdef USE_CUDNN
#include "caffe/util/cudnn.hpp"
#endif

//
// HIP macros
//

// HIP: various checks for different function calls.
#define HIP_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
       hipError_t error  = condition;\
       if (error != hipSuccess) { \
          fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
          exit(EXIT_FAILURE);\
       }\
     }while (0)

#define HIPBLAS_CHECK(condition) \
  do { \
    hipblasStatus_t status = condition; \
    CHECK_EQ(status, HIPBLAS_STATUS_SUCCESS) << " " \
      << caffe::hipblasGetErrorString(status); \
  } while (0)

// TODO: Get HIP equivalent
/*#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status); \
  } while (0)
*/

// HIP: grid stride looping
#define HIP_KERNEL_LOOP(i, n) \
  for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; \
       i < (n); \
       i += hipBlockDim_x * hipGridDim_x)

// HIP: check for error after kernel execution and exit loudly if there is one.
//TODO: Get HIP equivalent
//#define HIP_POST_KERNEL_CHECK HIP_CHECK(cudaPeekAtLastError())

namespace caffe {

// HIP: library error reporting.
const char* hipblasGetErrorString(hipblasStatus_t error);
// HIP: use 512 threads per block

#ifdef __HIP_PLATFORM_NVCC__
const int CAFFE_HIP_NUM_THREADS = 512;
#endif
#ifdef __HIP_PLATFORM_HCC__
const int CAFFE_HIP_NUM_THREADS = 256;
#endif

// HIP: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_HIP_NUM_THREADS - 1) / CAFFE_HIP_NUM_THREADS;
}

}  // namespace caffe

#endif  // CPU_ONLY

#endif  // CAFFE_UTIL_DEVICE_ALTERNATE_H_
