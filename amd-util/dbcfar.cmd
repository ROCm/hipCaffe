#/bin/bash
export HIP_DB=1
gdb --args build/tools/caffe train -solver=examples/cifar10/cifar10_quick_solver.prototxt

