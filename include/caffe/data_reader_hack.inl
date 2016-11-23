// HACK for initializing DataReader::bodies_

#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

  using boost::weak_ptr;

  map<const string, weak_ptr<DataReader::Body> > DataReader::bodies_;
} // namespace caffe
