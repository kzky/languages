import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

import time
import os
import argparse

def main(args):
    #os.environ["CUDA_​VISIBLE_​DEVICES"] = args.device_id
    #cuda.init()

    # Setting
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    EXPLICIT_BATCH = args.batch_size << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    trt_dtype = eval("trt.{}".format(args.datatype))
    np_dtype = np.float32 #eval("np.{}".format(args.datatype))
    max_batch_size = args.max_batch_size
    max_workspace_size = 1 << 20 # TODO: what is the unit?

    # Builder
    with trt.Builder(TRT_LOGGER) as builder:
        builder.max_batch_size = max_batch_size
        builder.max_workspace_size = max_workspace_size
        builder.fp16_mode = True if trt_dtype == trt.float16 else False
        builder.int8_mode = True if trt_dtype == trt.int8 else False
        # Network
        with builder.create_network(EXPLICIT_BATCH) as network:
            # Parse ONNX
            parser = trt.OnnxParser(network, TRT_LOGGER)
            with open(args.onnx, "rb") as model:
                ret = parser.parse(model.read())
            if ret is False:
                TRT_LOGGER.log(trt.Logger.INFO, "ONNX parser error: {}".format(parser.get_error(0)))
                raise RuntimeError("ONNX parse error")

            TRT_LOGGER.log(trt.Logger.INFO, "Num Inputs: {}".format(network.num_inputs))
            TRT_LOGGER.log(trt.Logger.INFO, "Num Outputs: {}".format(network.num_outputs))
            TRT_LOGGER.log(trt.Logger.INFO, "Num Layers: {}".format(network.num_layers))

            TRT_LOGGER.log(trt.Logger.INFO, "\n===== Layers =====")
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                print("layer type: {}".format(layer.type))
                print("layer name: {}".format(layer.name))
            TRT_LOGGER.log(trt.Logger.INFO, "\n==================")

            # (Fake) dynamic range if dtype = int8
            if trt_dtype == trt.int8:
                for i in range(network.num_layers):
                    layer = network.get_layer(i)
                    for j in range(layer.num_inputs):
                        layer.get_input(j).dynamic_range = (-1.0, 1.0)
                        layer.get_input(j).dtype = trt_dtype
                for i in range(network.num_outputs):
                    network.get_output(i).dynamic_range = (-1.0, 1.0)
                    network.get_output(i).dtype = trt_dtype
            
            # Config
            with builder.create_builder_config() as config:
                # Runtime
                runtime = trt.Runtime(TRT_LOGGER)
                # Engine
                et_opt = 0
                if not args.trt:
                    st = time.time()
                    engine = builder.build_cuda_engine(network)
                    et_opt = time.time() - st
                    #engine = builder.build_engine(network, config)
                    # Serialize
                    network_name = "{}-{}.trt".format(args.onnx.split(".onnx")[0], args.datatype)
                    with open(network_name, "wb") as fp:
                        fp.write(engine.serialize())
                else:
                    with open(args.trt, "rb") as fp:
                        engine = runtime.deserialize_cuda_engine(fp.read())
                # Context
                with engine.create_execution_context() as context:
                    # Input/Output and Stream
                    h_inputs, h_outputs = [], []
                    d_inputs, d_outputs = [], []
                    for i in range(network.num_inputs):
                        h_input = cuda.pagelocked_empty(
                            trt.volume(engine.get_binding_shape(i)), dtype=np_dtype)
                        h_inputs.append(h_input)
                        d_input = cuda.mem_alloc(h_input.nbytes)
                        d_inputs.append(d_input)
                    for i in range(network.num_outputs):
                        h_output = cuda.pagelocked_empty(
                            trt.volume(engine.get_binding_shape(network.num_inputs + i)),
                            dtype=np_dtype)
                        d_output = cuda.mem_alloc(h_output.nbytes)
                        d_outputs.append(d_output)                        
                    stream = cuda.Stream()                    
                    for d_input, h_input in zip(d_inputs, h_inputs):
                        cuda.memcpy_htod_async(d_input, h_input, stream)
                    for d_output, h_output in zip(d_outputs, h_outputs):
                        cuda.memcpy_htod_async(d_output, h_output, stream)
                    bindings = [int(data) for data in d_inputs + d_outputs]
                    # Warmup
                    for _ in range(args.n_warmup_iter):
                        context.execute_async(bindings=bindings, stream_handle=stream.handle)
                    #stream.synchronize()
                    cuda.Context.synchronize()
                    # Bench only for a compute part
                    et_lat = []
                    for i in range(args.n_iter):
                        st = time.time()
                        context.execute_async(bindings=bindings, stream_handle=stream.handle)
                        #stream.synchronize()
                        cuda.Context.synchronize()
                        et_lat.append(time.time() - st)
                    print("Engine Device Memory = {}".format(engine.device_memory_size))
                    print("Optimization Time [s] = {}".format(et_opt))
                    print("Ave [ms] = {}".format(np.mean(et_lat) * 1000))
                    print("Std [ms] = {}".format(np.std(et_lat) * 1000))
                    print("Med [ms] = {}".format(np.median(et_lat) * 1000))
                    print("Min [ms] = {}".format(np.min(et_lat) * 1000))
                    print("Max [ms] = {}".format(np.max(et_lat) * 1000))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--trt", type=str)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--n-warmup-iter", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--datatype", type=str, default="float32",
                        choices=["float32", "float16", "int8"])
    parser.add_argument("--device-id", type=str, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
