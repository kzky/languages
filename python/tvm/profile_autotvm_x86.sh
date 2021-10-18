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
target=$2

num_threads=$(echo $nthreads | cut -d "-" -f 2)
num_threads=$((${num_threads} + 1))
for network in "${networks[@]}"; do
    command="python3 autotune_x86.py --onnx ../benchmark_nets/${network}.onnx --target \"${target}\" --nthreads \"${num_threads}\""
    file="${network}-\"${target}\"-${nthreads}.out"
    bash -c "docker run --rm -v ${HOME}:${HOME} -w $(pwd) --cpuset-cpus ${nthreads} tvmai/ci-cpu:v0.62-rebuild ${command} 2>&1 | tee ${file}"
    file="${network}-${target}-${nthreads}.out"
    mv "${file}" $(ls "${file}" | sed "s/ //g")
done
