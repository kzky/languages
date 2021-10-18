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

target=$1
dtype=$2

for network in "${networks[@]}"; do
    command="python3 profile_tvm.py --onnx ../benchmark_nets/${network}.onnx --target \"${target}\" --dtype ${dtype}"
    file="${network}-\"${target}\"-${dtype}.out"
    bash -c "nvidia-docker run --rm -v ${HOME}:${HOME} -w $(pwd) tvmai/ci-gpu:v0.64-rebuild ${command} 2>&1 | tee ${file}"
    file="${network}-${target}-${dtype}.out"
    mv "${file}" $(ls "${file}" | sed "s/ //g")
done
