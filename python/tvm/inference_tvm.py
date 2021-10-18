import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_runtime

import argparse

def main(args):
    # ONNX
    onnx_model = onnx.load(args.onnx)

    # Input shape (TODO: address multiple inputs)
    inp_dims = onnx_model.graph.input[-1].type.tensor_type.shape.dim
    input_name = onnx_model.graph.input[-1].name
    inp_shape = [dim.dim_value for dim in inp_dims]
    out_dims = onnx_model.graph.output[-1].type.tensor_type.shape.dim
    out_shape = [dim.dim_value for dim in out_dims]
    
    # Compile
    target = args.target
    shape_dict = {input_name: inp_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    ctx = tvm.context(target, args.device_id)
    with relay.build_config(opt_level=2):
        graph, lib, params = relay.build(mod, target, params=params)
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**params)

    # Execute
    dtype = args.dtype
    x = np.random.randn(*inp_shape).astype(dtype)
    tx = tvm.nd.array(x)
    module.set_input(input_name, tx)
    module.run()
    print(module.get_output(0, tvm.nd.empty(out_shape)).asnumpy().shape)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TVM ONNX Inference')
    parser.add_argument('--onnx', type=str, required=True)
    parser.add_argument('--target', type=str, required=True,
                        help="'llvm' for the naive cpu, " \
                        "'llvm -mcpu=core-avx2' for core-i7, " \
                        "'llvm -mcpu=skylake-avx512' for aws c5." \
                        "'cuda -libs=cudnn' for nvidia gpus")
    parser.add_argument('--dtype', type=str, default="float32")
    parser.add_argument('--device-id', type=int, default=0)
    args = parser.parse_args()
    main(args)
