# TVM related

## Doc
- https://docs.tvm.ai/index.html
- https://docs.tvm.ai/tutorials/frontend/from_onnx.html#sphx-glr-tutorials-frontend-from-onnx-py
- https://docs.tvm.ai/tutorials/index.html#auto-tuning

### Github
- https://github.com/apache/incubator-tvm

## DockerHub
- https://hub.docker.com/r/tvmai/


## Docker

For CPU, 

```bash
docker build  -t "tvmai/ci-cpu:v0.62-rebuild" -f ./docker/Dockerfile.cpu .
```

If the build of tvm fails, check the docker/Dockerfile.cpu, then change the `sed` command parts to specify the version of LLVM directly.

For CUDA GPU, 

```bash
docker build  -t "tvmai/ci-gpu:v0.64-rebuild" -f ./docker/Dockerfile.cuda .
```


## Run Profiling

For profiling all networks for TVM do the follows. For each network see 
a corresponding python script.

### Create NNP and ONNX

See [the common case](../benchmark_nets/README.md).

### Profile with non-AutoTVM

Before running the script, check and change `target` in the profile_tvm.py according to the target device like

#### x86 CPU

By not using the autotune, 

```bash
bash profile_tvm_x86.sh <cpuset-cpus> <target>
```

`<cpuset-cpus>` is the docker option to limit the number of cpus used. Set it as like `0-3`, meaning use 4 cpus.
'<target>' is either of 
    - 'llvm -mcpu=core-avx2' # core-i7
    - 'llvm -mcpu=skylake-avx512' # aws c5
    - 'cuda' # nvidia gpu
    
By using the autotune, 

```bash
bash profile_autotvm_x86.sh <cpuset-cpus> <target>
```

`<cpuset-cpus>` is the docker option to limit the number of cpus used. Set it as like `0-3`, meaning use 4 cpus.
'<target>' is either of 
    - 'llvm -mcpu=core-avx2' # core-i7
    - 'llvm -mcpu=skylake-avx512' # aws c5
    - 'cuda' # nvidia gpu


#### CUDA

By not using the autotune, 

```bash
bash profile_tvm_cuda.sh <target> <dtype>
```

'<target>' is either of 
    - 'cuda'
    - 'cuda -libs=cudnn, cublas'
'<dtype>' is either of 
    - 'float32'
    - 'float16'

By using the autotune, 

```bash
bash profile_autotvm_cuda.sh <target> <dtype>
```

'<target>' is either of 
    - 'cuda'
    - 'cuda -libs=cudnn, cublas'
'<dtype>' is of 
    - 'float32'

