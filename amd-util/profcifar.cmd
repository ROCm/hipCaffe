#/bin/bash
rocm-profiler -o ./cifar10.atp -w . -A  -e HIP_PROFILE_API=1  build/tools/caffe train --solver=examples/cifar10/cifar10_quick_solver.prototxt
$HIP_PATH/bin/hipdemangleatp ./cifar10.atp

