#include "caffe/caffe.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
#ifndef CPU_ONLY
  hipDeviceProp_t CAFFE_TEST_HIP_PROP;
#endif
}

#ifndef CPU_ONLY
using caffe::CAFFE_TEST_HIP_PROP;
#endif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  caffe::GlobalInit(&argc, &argv);
#ifndef CPU_ONLY
  // Before starting testing, let's first print out a few hip defice info.
  int device;
  hipGetDeviceCount(&device);
  cout << "Cuda number of devices: " << device << endl;
  if (argc > 1) {
    // Use the given device
    device = atoi(argv[1]);
    hipSetDevice(device);
    cout << "Setting to use device " << device << endl;
  } else if (HIP_TEST_DEVICE >= 0) {
    // Use the device assigned in build configuration; but with a lower priority
    device = HIP_TEST_DEVICE;
  }
  hipGetDevice(&device);
  cout << "Current device id: " << device << endl;
  hipGetDeviceProperties(&CAFFE_TEST_HIP_PROP, device);
  cout << "Current device name: " << CAFFE_TEST_HIP_PROP.name << endl;
#endif
  // invoke the test.
  return RUN_ALL_TESTS();
}
