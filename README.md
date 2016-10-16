# ** HIP backend Implementation for Caffe ** #


##Introduction: ##

This repository hosts the HIP backend implementation project for  [Caffe](https://github.com/BVLC/caffe). To know what HIP is please refer [here](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP). Caffe framework currently has a CUDA backend support targeting NVidia devices.  The goal of this project is to develop  HIP based codes targeting modern AMD devices. This project mainly targets the linux platform 

##Prerequisites: ##

**Hardware Requirements:**

* CPU: mainstream brand, Better if with >=4 Cores Intel Haswell based CPU 
* System Memory >= 4GB (Better if >10GB for NN application over multiple GPUs)
* Hard Drive > 200GB (Better if SSD or NVMe driver  for NN application over multiple GPUs)
* Minimum GPU Memory (Global) > 2GB

**GPU SDK and driver Requirements:**

* dGPUs: AMD R9 Fury X, R9 Fury, R9 Nano
* APUs: AMD APU Kaveri or Carrizo

**System software requirements:**

* Ubuntu 14.04 trusty
* GCC 4.6 and later
* CPP 4.6 and later (come with GCC package)
* python 2.7 and later
* HCC 0.9 from [here](https://bitbucket.org/multicoreware/hcc/downloads/hcc-0.9.16041-0be508d-ff03947-5a1009a-Linux.deb)


**Tools and Misc Requirements:**

* git 1.9 and later
* cmake 2.6 and later (2.6 and 2.8 are tested)


## Tested Environment so far: 

This section enumerates the list of tested combinations of Hardware and system software

**GPU Cards tested:**

* Radeon R9 Nano
* Radeon R9 FuryX 
* Radeon R9 Fury 
* Kaveri and Carizo APU

**Driver versions tested**  

* Boltzmann Early Release Driver for dGPU

   ROCM 1.2 Release : https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md
     
* Traditional HSA driver for APU (Kaveri)

**Desktop System Tested**

* Supermicro SYS-7048GR-TR  Tower 4 R9 Nano
* ASUS X99-E WS motherboard with 4 R9 Nano
* Gigabyte GA-X79S 2 AMD R9 Nano

**Server System Tested**

* Supermicro SYS 2028GR-THT  6 R9 NANO
* Supermicro SYS-1028GQ-TRT 4 R9 NANO
* Supermicro SYS-7048GR-TR Tower 4 R9 NANO
 

## Installation Flow: 

A. ROCM 1.2 Installation (If not done so far)

B. Pre-requisites Installation

C. HipCaffe Build

D. Unit Testing


## Installation Steps in detail:

### A. ROCM 1.2 Installation: 

  To Know more about ROCM  refer https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md

  a. Installing Debian ROCM repositories
     
  Before proceeding, make sure to completely uninstall any pre-release ROCm packages
     
  Refer https://github.com/RadeonOpenCompute/ROCm#removing-pre-release-packages for instructions to remove pre-release ROCM packages
     
  Steps to install rocm package are 
     
      * wget -qO - http://packages.amd.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
      
      * sudo sh -c 'echo deb [arch=amd64] http://packages.amd.com/rocm/apt/debian/ trusty main > /etc/apt/sources.list.d/rocm.list'
     
      * sudo apt-get update
      
      * sudo apt-get install rocm
      
      * Reboot the system
      
  b. Once Reboot, verify the installation
    
  To verify that the ROCm stack completed successfully you can execute to HSA vector_copy sample application:

       * cd /opt/rocm/hsa/sample
        
       * make
       
       * ./vector_copy

### B. Pre-requisites Installation: 

a. Support libraries 

          sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev libblas-dev libgflags-dev libgoogle-glog-dev liblmdb-dev libboost-all-dev

b. HcBLAS library
 
        *  wget https://bitbucket.org/multicoreware/hipcaffe/downloads/hcblas-hipblas-0c1e60d-Linux.deb

        * sudo dpkg -i hcblas-hipblas-0c1e60d-Linux.deb
 (hcblas gets installed under /opt/rocm/hcblas path)

      
### C. Hicaffe Build Steps:
  
    * make 

    * make test

To improve build time, one could as well invoke make -j <number of threads>


## D. Unit Testing ##

After done with A, B and C, Now its time to test. Run the following commands to perform unit testing of different components of Caffe.

             ./build/test/test_all.testbin