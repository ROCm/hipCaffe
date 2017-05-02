# HIP backend Implementation for Caffe #


## Introduction ##

This repository hosts the HIP backend implementation project for  [Caffe](https://github.com/BVLC/caffe). To know what HIP is please refer [here](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP). Caffe framework currently has a CUDA backend support targeting NVidia devices.  The goal of this project is to develop  HIP based codes targeting modern AMD devices. This project mainly targets the linux platform 

## Prerequisites ##

### Hardware Requirements ###

* For ROCm hardware requirements, see [here](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md#supported-cpus)

### Software and Driver Requirements ###

* For ROCm software requirements, see [here](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md#the-latest-rocm-platform---rocm-15)

## Installation Flow ##

A. ROCm Installation (If not done so far)

B. Pre-requisites Installation

C. hipCaffe Build

D. Unit Testing

E. Simple Workload Examples


## Installation Steps in Detail ##

### A. ROCm Installation ##

  To Know more about ROCM  refer https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md

  a. Installing Debian ROCM repositories
     
  Before proceeding, make sure to completely uninstall any pre-release ROCm packages
     
  Refer https://github.com/RadeonOpenCompute/ROCm#removing-pre-release-packages for instructions to remove pre-release ROCM packages
     
  Steps to install rocm package are 
     
      * wget -qO - http://packages.amd.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
      
      * sudo sh -c 'echo deb [arch=amd64] http://packages.amd.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'
     
      * sudo apt-get update
      
      * sudo apt-get install rocm rocm-opencl rocm-opencl-dev
      
      * Reboot the system
      
  b. Then, verify the installation

  Double-check your kernel (at a minimum, you should see "kfd" in the name):

      * uname -r

  To verify that the ROCm stack completed successfully you can execute to HSA vector_copy sample application:

      * cd /opt/rocm/hsa/sample
        
      * make
       
      * ./vector_copy

### B. Pre-requisites Installation ###

a. Support libraries 

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

b. ROCm libraries

       *  wget https://bitbucket.org/multicoreware/hipcaffe/downloads/hcblas-hipblas-0c1e60d-Linux.deb

       * sudo dpkg -i hcblas-hipblas-0c1e60d-Linux.deb
 (hcblas gets installed under /opt/rocm/hcblas path)

      
### C. hipCaffe Build Steps ###
  
       * make 

       * make test

To improve build time, one could as well invoke make -j <number of threads>


## D. Unit Testing ##

After done with A, B and C, Now its time to test. Run the following commands to perform unit testing of different components of Caffe.

       * ./build/test/test_all.testbin

## E. Example Workloads ##

### MNIST training ###

Steps:

       * ./data/mnist/get_mnist.sh

       * ./examples/mnist/create_mnist.sh
       
       * ./examples/mnist/train_lenet.sh

### CIFAR-10 training ###

Steps:  

       * ./data/cifar10/get_cifar10.sh
       
       * ./examples/cifar10/create_cifar10.sh
       
       * ./build/tools/caffe train --solver=examples/cifar10/cifar10_quick_solver.prototxt
       