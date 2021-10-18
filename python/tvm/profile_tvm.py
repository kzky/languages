import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_runtime

import argparse
import time

def main(args):
    # ONNX
    onnx_model = onnx.load(args.onnx)

    # Input shape (TODO: address multiple inputs)
    dims = onnx_model.graph.input[-1].type.tensor_type.shape.dim
    input_name = onnx_model.graph.input[-1].name
    inp_shape = [dim.dim_value for dim in dims]

    # Compile
    target = args.target    
    shape_dict = {input_name: inp_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    ctx = tvm.context(target, args.device_id)
    with relay.build_config(opt_level=2):
        graph, lib, params = relay.build(mod, target, params=params)

    # Evaluate
    dtype = args.dtype
    x = np.random.randn(*inp_shape).astype(dtype)
    tx = tvm.nd.array(x)
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(input_name, tx) # this gives 1 HtoD copy
    module.set_input(**params)       # this gives 54 HtoD copy, but there is another HtoD copy
    
    # Profile
    for i in range(args.warmup_niter):
        module.run()
    latencies = []
    for i in range(args.niter):
        st = time.time()
        module.run()
        latencies.append(time.time() - st)
    print("Ave [ms] = {}".format(np.mean(latencies) * 1000))
    print("Std [ms] = {}".format(np.std(latencies) * 1000))
    print("Med [ms] = {}".format(np.median(latencies) * 1000))
    print("Min [ms] = {}".format(np.min(latencies) * 1000))
    print("Max [ms] = {}".format(np.max(latencies) * 1000))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TVM ONNX Latency Measurement.')
    parser.add_argument('--onnx', type=str, required=True)
    parser.add_argument('--target', type=str, required=True,
                        help="'llvm' for the naive cpu, " \
                        "'llvm -mcpu=core-avx2' for core-i7, " \
                        "'llvm -mcpu=skylake-avx512' for aws c5." \
                        "'cuda -libs=cudnn' for nvidia gpus")
    parser.add_argument('--dtype', type=str, default="float32")
    parser.add_argument('--warmup-niter', type=int, default=10)
    parser.add_argument('--niter', type=int, default=100)
    parser.add_argument('--device-id', type=int, default=0)
    
    args = parser.parse_args()

    main(args)
