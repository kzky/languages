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
    #"SAGAN"
)

cmd=/usr/src/tensorrt/bin/trtexec
opts=$@

for network in "${networks[@]}"; do
    sudo $cmd  --onnx=../benchmark_nets/${network}.onnx ${opts} 2>&1 | tee "${network}-${opts}-trt.out"
done

