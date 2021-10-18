"""
This example follows the classification_sample.py in OpenVINO example
"""


from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. " \
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the" \
                           " kernels implementations.", type=str, default=None)
    args.add_argument('-b', '--batch_size', type=int, required=False, default=1)
    args.add_argument('-nthreads', '--number_threads', type=int, required=False, default=1,
                      help='Number of threads to use for inference on the CPU ' \
                      '(including HETERO and MULTI cases).')
    return parser


def main():
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    device = "CPU"

    # Inference Engine
    ie = IECore()
    ie.set_config({'CPU_THREADS_NUM': str(args.number_threads)}, device)
    ie.add_extension(args.cpu_extension, device) if args.cpu_extension else None
    
    
    # Read IR
    net = ie.read_network(model=model_xml, weights=model_bin)

    # Check supported layers
    supported_layers = ie.query_network(net, device)
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
        print("Following layers are not supported by the plugin for specified device {}:\n {}".
              format(args.device, ', '.join(not_supported_layers)))
        print("Please try to specify cpu extensions library path using -l " \
              "or --cpu_extension command line argument.")
        raise RuntimeError("Some layers are not supported.")

    assert len(net.inputs.keys()) == 1, "Sample uspports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    # Network inputs and outputs
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = args.batch_size

    # Read and pre-process input images (default is BGR order)
    n, c, h, w = net.inputs[input_blob].shape
    images = np.random.randn(*[n, c, h, w])

    # Executable network
    exec_net = ie.load_network(network=net, device_name=device)

    # Sync inference
    res = exec_net.infer(inputs={input_blob: images})

    print(list(res.values())[0].shape)

if __name__ == '__main__':
    sys.exit(main() or 0)
