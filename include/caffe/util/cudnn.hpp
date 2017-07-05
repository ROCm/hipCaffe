#ifndef CAFFE_UTIL_CUDNN_H_
#define CAFFE_UTIL_CUDNN_H_
#ifdef USE_ACCMI

#ifdef USE_MIOPEN
#include <miopen/miopen.h>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#define MIOPEN_CHECK(condition) \
  do { \
    miopenStatus_t status = condition; \
    CHECK_EQ(status, miopenStatusSuccess) << " "\
      << miopenGetErrorString(status); \
  } while (0)

inline const char* miopenGetErrorString(miopenStatus_t status) {
  switch (status) {
  case miopenStatusSuccess:
    return "miopenStatusSuccess";
  case miopenStatusNotInitialized:
    return "miopenStatusNotInitialized";
  case miopenStatusInvalidValue:
    return "miopenStatusInvalidValue";
  case miopenStatusBadParm:
    return "miopenStatusBadParm";
  case miopenStatusAllocFailed:
    return "miopenStatusAllocFailed";
  case miopenStatusInternalError:
    return "miopenStatusInternalError";
  case miopenStatusNotImplemented:
    return "miopenStatusNotImplemented";
  case miopenStatusUnknownError:
    return "miopenStatusUnknownError";
  }
  return "Unknown MIOpen status";
}

namespace caffe {

namespace miopen {

template <typename Dtype> class dataType;
template<> class dataType<float>  {
 public:
  static const miopenDataType_t type = miopenFloat;
  static float oneval, zeroval;
  static const void *one, *zero;
};
template<> class dataType<double> {
 public:
  dataType<double>() { assert(0); };
  static const miopenDataType_t type = miopenFloat; //TODO-miopenDouble;
  static double oneval, zeroval;
  static const void *one, *zero;
};

template <typename Dtype>
inline void createTensor4dDesc(miopenTensorDescriptor_t* desc) {
  MIOPEN_CHECK(miopenCreateTensorDescriptor(desc));
}

template <typename Dtype>
inline void setTensor4dDesc(miopenTensorDescriptor_t* desc,
    int n, int c, int h, int w,
    int stride_n, int stride_c, int stride_h, int stride_w) {
  // MIOpen doesn't API to set stride_n, stride_c, stride_h, stride_w yet
  MIOPEN_CHECK(miopenSet4dTensorDescriptor(*desc, dataType<Dtype>::type,
        n, c, h, w));
}

template <typename Dtype>
inline void setTensor4dDesc(miopenTensorDescriptor_t* desc,
    int n, int c, int h, int w) {
  const int stride_w = 1;
  const int stride_h = w * stride_w;
  const int stride_c = h * stride_h;
  const int stride_n = c * stride_c;
  setTensor4dDesc<Dtype>(desc, n, c, h, w,
                         stride_n, stride_c, stride_h, stride_w);
}

template <typename Dtype>
inline void createFilterDesc(miopenTensorDescriptor_t* desc,
    int n, int c, int h, int w) {
  MIOPEN_CHECK(miopenCreateTensorDescriptor(desc));
  MIOPEN_CHECK(miopenSet4dTensorDescriptor(*desc, dataType<Dtype>::type,
        n, c, h, w));
}

template <typename Dtype>
inline void createConvolutionDesc(miopenConvolutionDescriptor_t* conv) {
  MIOPEN_CHECK(miopenCreateConvolutionDescriptor(conv));
}

template <typename Dtype>
inline void setConvolutionDesc(miopenConvolutionDescriptor_t* conv,
    miopenTensorDescriptor_t bottom, miopenTensorDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  MIOPEN_CHECK(miopenInitConvolutionDescriptor(*conv, miopenConvolution,
        pad_h, pad_w, stride_h, stride_w, 1, 1));
}

template <typename Dtype>
inline void createPoolingDesc(miopenPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, miopenPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
  switch (poolmethod) {
  case PoolingParameter_PoolMethod_MAX:
    *mode = miopenPoolingMax;
    break;
  case PoolingParameter_PoolMethod_AVE:
    *mode = miopenPoolingAverage;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  MIOPEN_CHECK(miopenCreatePoolingDescriptor(pool_desc));
  MIOPEN_CHECK(miopenSet2dPoolingDescriptor(*pool_desc, *mode,
        h, w, pad_h, pad_w, stride_h, stride_w));
}

template <typename Dtype>
inline void createActivationDescriptor(miopenActivationDescriptor_t* activ_desc,
    miopenActivationMode_t mode) {
  MIOPEN_CHECK(miopenCreateActivationDescriptor(activ_desc));
  MIOPEN_CHECK(miopenSetActivationDescriptor(*activ_desc, mode,
                                           Dtype(0), Dtype(0), Dtype(0)));
}

}  // namespace miopen

}  // namespace caffe

#endif

#ifdef USE_CUDNN

#include <cudnn.h>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#define CUDNN_VERSION_MIN(major, minor, patch) \
    (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " "\
      << cudnnGetErrorString(status); \
  } while (0)

inline const char* cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
  }
  return "Unknown cudnn status";
}

namespace caffe {

namespace cudnn {

template <typename Dtype> class dataType;
template<> class dataType<float>  {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
};
template<> class dataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
};

template <typename Dtype>
inline void createTensor4dDesc(cudnnTensorDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
    int n, int c, int h, int w,
    int stride_n, int stride_c, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, dataType<Dtype>::type,
        n, c, h, w, stride_n, stride_c, stride_h, stride_w));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
    int n, int c, int h, int w) {
  const int stride_w = 1;
  const int stride_h = w * stride_w;
  const int stride_c = h * stride_h;
  const int stride_n = c * stride_c;
  setTensor4dDesc<Dtype>(desc, n, c, h, w,
                         stride_n, stride_c, stride_h, stride_w);
}

template <typename Dtype>
inline void createFilterDesc(cudnnFilterDescriptor_t* desc,
    int n, int c, int h, int w) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
      CUDNN_TENSOR_NCHW, n, c, h, w));
#else
  CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(*desc, dataType<Dtype>::type,
      CUDNN_TENSOR_NCHW, n, c, h, w));
#endif
}

template <typename Dtype>
inline void createConvolutionDesc(cudnnConvolutionDescriptor_t* conv) {
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv));
}

template <typename Dtype>
inline void setConvolutionDesc(cudnnConvolutionDescriptor_t* conv,
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
      pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION));
}

template <typename Dtype>
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
  switch (poolmethod) {
  case PoolingParameter_PoolMethod_MAX:
    *mode = CUDNN_POOLING_MAX;
    break;
  case PoolingParameter_PoolMethod_AVE:
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, *mode,
        CUDNN_PROPAGATE_NAN, h, w, pad_h, pad_w, stride_h, stride_w));
#else
  CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(*pool_desc, *mode,
        CUDNN_PROPAGATE_NAN, h, w, pad_h, pad_w, stride_h, stride_w));
#endif
}

template <typename Dtype>
inline void createActivationDescriptor(cudnnActivationDescriptor_t* activ_desc,
    cudnnActivationMode_t mode) {
  CUDNN_CHECK(cudnnCreateActivationDescriptor(activ_desc));
  CUDNN_CHECK(cudnnSetActivationDescriptor(*activ_desc, mode,
                                           CUDNN_PROPAGATE_NAN, Dtype(0)));
}

}  // namespace cudnn

}  // namespace caffe

#endif  // USE_CUDNN
#endif // USE_ACCMI
#endif  // CAFFE_UTIL_CUDNN_H_
