# hipCaffe: the HIP Port of Caffe #


## Introduction ##

This repository hosts the HIP port of [Caffe](https://github.com/BVLC/caffe) (or hipCaffe, for short). For details on HIP, please refer [here](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP). This HIP-ported framework is able to target both AMD ROCm and Nvidia CUDA devices from the same source code. Hardware-specific optimized library calls are also supported within this codebase.

## Prerequisites ##

### Hardware Requirements ###

* For ROCm hardware requirements, see [here](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md#supported-cpus)

### Software and Driver Requirements ###

* For ROCm software requirements, see [here](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md#the-latest-rocm-platform---rocm-15)

## Installation ##

### AMD ROCm Installation ###

For further background information on ROCm, refer [here](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md)

Installing ROCm Debian packages:  
  
      wget -qO - http://packages.amd.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
      
      sudo sh -c 'echo deb [arch=amd64] http://packages.amd.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'
     
      sudo apt-get update
      
      sudo apt-get install rocm rocm-opencl rocm-opencl-dev
      
      sudo reboot

Then, verify the installation. Double-check your kernel (at a minimum, you should see "kfd" in the name):

      uname -r

In addition, check that you can run the simple HSA vector_copy sample application:

      cd /opt/rocm/hsa/sample
        
      make
       
      ./vector_copy

### Pre-requisites Installation ###

Install Caffe dependencies:

    apt-get update && apt-get install \
    	pkg-config \
    	protobuf-compiler \
    	libprotobuf-dev \
    	libleveldb-dev \
    	libsnappy-dev \
    	libhdf5-serial-dev \
    	libatlas-base-dev \
    	libboost-all-dev \
    	libgflags-dev \
    	libgoogle-glog-dev \
    	liblmdb-dev \
    	python-numpy python-scipy python3-dev python-yaml python-pip \
    	libopencv-dev \
    	libfftw3-dev \
    	libelf-dev

Install the necessary ROCm compute libraries:  


    sudo apt-get install rocm-libs

Add a few symbolic links:

    cd /opt/rocm/hcrng/lib  && sudo ln -s libhiprng_hcc.so  libhiprng.so

    cd /opt/rocm/hcblas/lib && sudo ln -s libhipblas_hcc.so libhipblas.so

      
### hipCaffe Build Steps ###

You may need to modify the Makefile.config file for your own installation.  Then, build it:  
  
    make 

To improve build time, consider invoking parallel make with the "-j$(nproc)" flag.


## Unit Testing ##

Run the following commands to perform unit testing of different components of Caffe.

    make test
    
    ./build/test/test_all.testbin

## Example Workloads ##

### MNIST training ###

Steps:

       ./data/mnist/get_mnist.sh

       ./examples/mnist/create_mnist.sh
       
       ./examples/mnist/train_lenet.sh

### CIFAR-10 training ###

Steps:  

       ./data/cifar10/get_cifar10.sh
       
       ./examples/cifar10/create_cifar10.sh
       
       ./build/tools/caffe train --solver=examples/cifar10/cifar10_quick_solver.prototxt
       