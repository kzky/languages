import os
import numpy as np
import onnx

import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_runtime as runtime
from tvm.contrib.util import tempdir

import argparse
import time

# You can skip the implementation of this function for this tutorial.
def tune_kernels(tasks,
                 measure_option,
                 n_trial=100,
                 tuner='gridsearch',
                 early_stopping=None,
                 log_filename='tuning.log'):

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        tsk_trial = min(n_trial, len(task.config_space))
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])

# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(graph, target, input_dict, records, opt_sch_file, use_DP=True):
    target_op = [relay.op.get("nn.conv2d"),]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, input_dict, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)
    
    
def main(args):
    # ONNX
    onnx_model = onnx.load(args.onnx)

    # Input shape (TODO: address multiple inputs)
    model_name = onnx_model.graph.name
    dims = onnx_model.graph.input[-1].type.tensor_type.shape.dim
    input_name = onnx_model.graph.input[-1].name
    inp_shape = [dim.dim_value for dim in dims]

    # Settings
    target = args.target
    log_file = "%s_%s_%s_tuning.log" % (model_name, "".join(args.target.split(" ")), args.nthreads)
    graph_opt_sch_file = "%s_%s_%s_graph_opt.log" % (model_name, "".join(args.target.split(" ")), args.nthreads)
    os.environ["TVM_NUM_THREADS"] = args.nthreads

    # mode, params
    shape_dict = {input_name: inp_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    # tuning option
    tuning_option = {
        'log_filename': log_file,
        'tuner': 'random',
        'early_stopping': None,
     
        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(number=10, repeat=1,
                                       min_repeat_ms=1000),
        ),
    }

    # task
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),))

    # run tuning tasks
    st = time.time()
    tune_kernels(tasks, **tuning_option)
    tune_graph(mod["main"], target, shape_dict, log_file, graph_opt_sch_file)
    print("Tune took [s] = {}".format(time.time() - st))

    # compile kernels with graph-level best records
    with autotvm.apply_graph_best(graph_opt_sch_file):
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # export library
        tmp = tempdir()
        filename = "{}.tar".format(model_name)
        lib.export_library(tmp.relpath(filename))

        # upload parameters to device
        ctx = tvm.context(target, args.device_id)
        dtype = args.dtype
        x = np.random.randn(*inp_shape).astype(dtype)
        module = runtime.create(graph, lib, ctx)
        module.set_input(input_name, tvm.nd.array(x))
        module.set_input(**params)

    # Profile
    for i in range(args.warmup_niter):
        tvm_output = module.run()
    latencies = []
    for i in range(args.niter):
        st = time.time()
        tvm_outputs = module.run()
        latencies.append(time.time() - st)
    print("Ave [ms] = {}".format(np.mean(latencies) * 1000))
    print("Std [ms] = {}".format(np.std(latencies) * 1000))
    print("Med [ms] = {}".format(np.median(latencies) * 1000))
    print("Min [ms] = {}".format(np.min(latencies) * 1000))
    print("Max [ms] = {}".format(np.max(latencies) * 1000))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto TVM ONNX.')
    parser.add_argument('--onnx', type=str, required=True)
    parser.add_argument('--target', type=str, required=True,
                        help="'llvm' for the naive cpu, " \
                        "'llvm -mcpu=core-avx2' for core-i7, " \
                        "'llvm -mcpu=skylake-avx512' for aws c5." \
                        "'cuda -libs=cudnn' for nvidia gpus")
    parser.add_argument('--dtype', type=str, default="float32")
    parser.add_argument('--warmup-niter', type=int, default=10)
    parser.add_argument('--niter', type=int, default=100)
    parser.add_argument('--nthreads', type=str, default="4")
    parser.add_argument('--device-id', type=int, default=0)
    args = parser.parse_args()
    main(args)
