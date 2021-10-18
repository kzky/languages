import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

import time
import argparse

def main(args):
    #cuda.init()

    # Setting
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    datatype = trt.float32
    max_batch_size = 1
    max_workspace_size = 1 << 20

    model_file = args.onnx

    # Builder
    with trt.Builder(TRT_LOGGER) as builder:
        builder.max_batch_size = max_batch_size
        builder.max_workspace_size = max_workspace_size
        builder.int8_mode = False
        builder.fp16_mode = False
        # Network
        with builder.create_network(EXPLICIT_BATCH) as network:
            parser = trt.OnnxParser(network, TRT_LOGGER)
            with open(model_file, "rb") as model:
                ret = parser.parse(model.read())
            TRT_LOGGER.log(trt.Logger.INFO, "ONNX parse result: {}".format(ret))
            TRT_LOGGER.log(trt.Logger.INFO, "ONNX parser error: {}".format(parser.get_error(0)))
            TRT_LOGGER.log(trt.Logger.INFO, "Num Inputs: {}".format(network.num_inputs))
            TRT_LOGGER.log(trt.Logger.INFO, "Num Outputs: {}".format(network.num_outputs))
            TRT_LOGGER.log(trt.Logger.INFO, "Num Layers: {}".format(network.num_layers))

            TRT_LOGGER.log(trt.Logger.INFO, "\n===== Layers =====")
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                print("type: {}".format(layer.type))
                print("name: {}".format(layer.name))
            TRT_LOGGER.log(trt.Logger.INFO, "\n==================")
            # Config
            with builder.create_builder_config() as config:
                # Engine
                engine = builder.build_cuda_engine(network)
                # TODO: Serialzize
                
                #engine = builder.build_engine(network, config)
                # Context
                with engine.create_execution_context() as context:
                    # Input/Output and Stream
                    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
                    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
                    d_input = cuda.mem_alloc(h_input.nbytes)
                    d_output = cuda.mem_alloc(h_output.nbytes)
                    stream = cuda.Stream()

                    # Inference
                    x = np.random.randn(*h_input.shape)
                    h_input[:] = x
                    cuda.memcpy_htod_async(d_input, h_input, stream)
                    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
                    cuda.memcpy_dtoh_async(h_output, d_output, stream)
                    stream.synchronize()
                    print(h_output[:])
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TensorRT Inference.')
    parser.add_argument('--onnx', type=str, required=True)
    args = parser.parse_args()
    main(args)
