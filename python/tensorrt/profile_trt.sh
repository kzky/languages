#!/usr/bin/env bash

networks=(
    "ResNet-18"
    "ResNet-50"
    "MobileNet-V1"
    "MobileNet-V2"
    "DeepLabV3Plus"
    "YOLOv2"
    "Openpose"
    "ESRGAN"
    "SAGAN"
)

datatype=$1

for network in "${networks[@]}"; do
    command="python profile_trt.py \
           --onnx ../benchmark_nets/${network}.onnx \
           --datatype ${datatype} \
           2>&1 | tee ${network}-${datatype}-trt.out"
    file="${network}-${datatype}-trt.out"
    bash -c "nvidia-docker run -w $(pwd) -v ${HOME}:${HOME} -it my_trt/tensorrt ${command} 2>&1 | tee ${file}"
done

