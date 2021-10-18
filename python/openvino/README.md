# OpenVINO related

## Doc

### Top/Installation(DockerFile)/Device
- https://docs.openvinotoolkit.org/
- https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html
- https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_docker_linux.html
- https://github.com/openvinotoolkit/openvino/blob/2020/build-instruction.md
- https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_Supported_Devices.html

### Opset/Layers
- https://docs.openvinotoolkit.org/latest/_docs_ops_opset.html
- https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html

### Convert/Bench
- https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model.html
- https://docs.openvinotoolkit.org/latest/_inference_engine_tools_benchmark_tool_README.html

### Quantization
- https://docs.openvinotoolkit.org/latest/_README.html

### Samples
- https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Samples_Overview.html
- https://github.com/openvinotoolkit/openvino/blob/2020/inference-engine/ie_bridges/python/

### API Refs
- https://docs.openvinotoolkit.org/latest/annotated.html
- https://docs.openvinotoolkit.org/latest/ie_python_api/annotated.html

### Github
- https://github.com/openvinotoolkit/openvino

### DockerHub
- https://hub.docker.com/u/openvino


## Docker

```bash
docker pull openvino/ubuntu18_dev
```

## Run Profiling

For profiling all networks for OpenVINO do the follows. For each network see 
a corresponding python script.

### Create NNP and ONNX

See [the common case](../benchmark_nets/README.md).


### Profile for OpenVINO

In (the docker container and ) this directory, 


```bash
bash convert_onnx_to_openvino.sh
```


```bash
bash profile_openvino.sh <nthreads>
```

`<nthreads>` is the number of threads.


For the single onnx file, do 

```bash
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
    --framework onnx \
    --input_model <network>.onnx
```


```bash
python3 /opt/intel/openvino/deployment_tools/tools/benchmark_tool/benchmark_app.py \
    -m <network>.xml \
    -d CPU \
    -api sync \
    -niter 10 \
    -b 1 -nthreads <nthreads>
```


