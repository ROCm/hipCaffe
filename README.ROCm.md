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

Install ROCm Debian packages:  
  
      PKG_REPO="http://repo.radeon.com/rocm/apt/debian/"
      
      wget -qO - $PKG_REPO/rocm.gpg.key | sudo apt-key add -
      
      sudo sh -c "echo deb [arch=amd64] $PKG_REPO xenial main > /etc/apt/sources.list.d/rocm.list"
     
      sudo apt-get update
      
      sudo apt-get install rocm rocm-utils rocm-opencl rocm-opencl-dev rocm-profiler cxlactivitylogger

Next, update your paths and reboot: 

      echo 'export PATH=/opt/rocm/bin:$PATH' >> $HOME/.bashrc
      
      echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> $HOME/.bashrc

      source $HOME/.bashrc
      
      sudo reboot

Then, verify the installation. Double-check your kernel (at a minimum, you should see "kfd" in the name):

      uname -r

In addition, check that you can run the simple HSA vector_copy sample application:

      pushd /opt/rocm/hsa/sample
        
      make
       
      ./vector_copy
      
      popd

### Pre-requisites Installation ###

Install Caffe dependencies:

    sudo apt-get install \
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
	
Install some misc development dependencies:  

    sudo apt-get install git wget

Install the necessary ROCm compute libraries:  

    sudo apt-get install rocm-libs miopen-hip miopengemm

      
### hipCaffe Build Steps ###

Clone hipCaffe:

    git clone https://github.com/ROCmSoftwarePlatform/hipCaffe.git

    cd hipCaffe

You may need to modify the Makefile.config file for your own installation.  Then, build it:

    cp ./Makefile.config.example ./Makefile.config
    
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

### CaffeNet inference ###

Steps:

       ./data/ilsvrc12/get_ilsvrc_aux.sh

       ./scripts/download_model_binary.py models/bvlc_reference_caffenet

       ./build/examples/cpp_classification/classification.bin \
            models/bvlc_reference_caffenet/deploy.prototxt \
	    models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
	    data/ilsvrc12/imagenet_mean.binaryproto \
	    data/ilsvrc12/synset_words.txt \
	    examples/images/cat.jpg

## Known Issues

### Temp workaround for multi-GPU data transfer error

Sometimes when training with multiple GPUs, we hit this type of error signature:  
```
*** SIGSEGV (@0x0) received by PID 57122 (TID 0x7fd841500b80) from PID 0; stack trace: ***
    @     0x7fd8409a1390 (unknown)
    @     0x7fd8400a71f7 (unknown)
    @     0x7fd840515263 (unknown)
    @     0x7fd81f5ef907 UnpinnedCopyEngine::CopyHostToDevice()
    @     0x7fd81f5d3bb9 HSACopy::syncCopyExt()
    @     0x7fd81f5d28bc Kalmar::HSAQueue::copy_ext()
    @     0x7fd8410dba5b ihipStream_t::locked_copySync()
    @     0x7fd8411030bf hipMemcpy
    @           0x6cfd43 caffe::caffe_gpu_rng_uniform()
    @           0x5a32ba caffe::DropoutLayer<>::Forward_gpu()
    @           0x430bbf caffe::Layer<>::Forward()
    @           0x6fefe7 caffe::Net<>::ForwardFromTo()
    @           0x6feeff caffe::Net<>::Forward()
    @           0x801e8c caffe::Solver<>::Step()
    @           0x8015c3 caffe::Solver<>::Solve()
    @           0x71a277 caffe::P2PSync<>::Run()
    @           0x42dcbc train()
```

See this [comment](https://github.com/ROCmSoftwarePlatform/hipCaffe/issues/11#issuecomment-318518802).

In short, here's the temporary workaround:  
```
export HCC_UNPINNED_COPY_MODE=2
```

Please note that we have a long-term solution -- using a new RNG lib -- that we'll be pushing out soon.  


## Tutorials

* [hipCaffe Quickstart Guide](https://rocm.github.io/ROCmHipCaffeQuickstart.html)
