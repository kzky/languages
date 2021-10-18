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

datatype=$1

for network in "${networks[@]}"; do
    python profile_trt5.py \
           --onnx ${network}.onnx \
           --datatype ${datatype} \
           --device-id 1 2>&1 | tee ${network}-${datatype}-trt.out
done

