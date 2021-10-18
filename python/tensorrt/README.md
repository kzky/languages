# TensorRT related


## Doc
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Onnx/pyOnnx.html
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-best-practices/index.html

### Github
- https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/trtexec

## PyCUDA
- https://documen.tician.de/pycuda/

## Jetson
- https://github.com/dusty-nv/jetson-inference

For pycuda installation

```bash
sudo python3 -m pip install --global-option=build_ext --global-option="-I/usr/local/cuda-10.0/targets/aarch64-linux/include/" --global-option="-L/usr/local/cuda-10.0/targets/aarch64-linux/lib/" pycuda
```

## Docker
- https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt (authentication required for Gaming GPU)

```bash
docker pull nvcr.io/nvidia/tensorrt:20.03-py3
```

or in the docker directory, 

```bash
docker build -t "my_trt/tensorrt" .
```

## Run Profiling

For profiling all networks for TensorRT do the follows. For each network see 
a corresponding python script.


### Create NNP and ONNX

See [the common case](../benchmark_nets/README.md).

### Profile for TensorRT

In this directory, 

```bash
bash profile_trt.sh <datatype>
```

`<datatype>` should either float32, float16, or int8.

For the single onnx file, do 

```bash
python profile_trt.py --onnx <network>.onnx 
```
