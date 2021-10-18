#!/usr/bin/env bash

networks=(
    "ResNet-18"
    "ResNet-50"
    "MobileNet-V1"
    "MobileNet-V2"
    "DeepLabV3Plus"
    "YOLOv2"
    "Openpose"
    #"ESRGAN"
    #"SAGAN"
)

nthreads=$1

source /opt/intel/openvino/bin/setupvars.sh

for network in "${networks[@]}"; do
    # Warmup
    python3 /opt/intel/openvino/deployment_tools/tools/benchmark_tool/benchmark_app.py \
            -m ${network}.xml \
            -d CPU \
            -api sync \
            -niter 10 \
            -b 1 -nthreads ${nthreads}
    # Bench
    python3 /opt/intel/openvino/deployment_tools/tools/benchmark_tool/benchmark_app.py \
            -m ${network}.xml \
            -d CPU \
            -api sync \
            -niter 100 \
            -b 1 -nthreads ${nthreads} 2>&1 | tee ${network}-openvino-bench-${nthreads}.out
done
