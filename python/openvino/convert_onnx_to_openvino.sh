#!/bin/bash

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

source /opt/intel/openvino/bin/setupvars.sh

for network in "${networks[@]}"; do
    python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
            --framework onnx \
            --input_model ../benchmark_nets/${network}.onnx 2>&1 | tee ${network}-openvino-mo.out
done

