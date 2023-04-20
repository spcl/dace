from typing import Tuple, Dict
import sys
sys.path.insert(0, '/apps/ault/spack/opt/spack/linux-centos8-zen/gcc-8.4.1/cuda-11.8.0-fjdnxm6yggxxp75sb62xrxxmeg4s24ml/nsight-compute-2022.3.0/extras/python/')
import ncu_report


def get_action(filename: str) -> ncu_report.IAction:
    """
    Gets the action inside the given ncu-rep file. Assumes there is only one range and action

    :param filename: The path/filename of the ncu-report
    :type filename: str
    """
    my_context = ncu_report.load_report(filename)
    num_ranges = my_context.num_ranges()
    if num_ranges > 1:
        print(f"WARNING: More than one range found in {filename} only taking the first")
    my_range = my_context.range_by_idx(0)
    num_actions = my_range.num_actions()
    if num_actions > 1:
        print(f"WARNING: More than one action found in {filename} only taking the first")
    my_action = my_range.action_by_idx(0)
    return my_action


def get_peak_performance(action: ncu_report.IAction) -> Tuple[float, float]:
    """
    Gets the peak performance given the ncu action.

    :param action: The ncu action object
    :type action: ncu_report.IAction
    :return: The peak performance and bandwidth in flop/s and byte/s
    :rtype: Tuple[float, float]
    """
    # bytes per memory cycle
    peak_beta = action.metric_by_name('dram__bytes.sum.peak_sustained').as_double()
    # memory cycles per second
    memory_frequency = action.metric_by_name('dram__cycles_elapsed.avg.per_second').as_double()
    # flop per gpu cycle
    pi = 2.0 * action.metric_by_name('sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained').as_double()
    # gpu cycles per second
    gpu_frequency = action.metric_by_name('gpc__cycles_elapsed.avg.per_second').as_double()
    return (pi * gpu_frequency, peak_beta * memory_frequency)


def get_achieved_work(action: ncu_report.IAction) -> Dict[str, int]:
    """
    Returns the amount of work achieved as a dictionary containg the amount of double and single precision adds, muls,
    fmas and work (dW, fW)
    :param action: The ncu action object
    :type action: ncu_report.IAction
    :return: Dictionary with all data, keys are: dadds, dmuls, dfmas, fadds, fmuls, ffmas, dW, fW. Values are in flop
    :rtype: Dict[str, int]
    """
    dadds = action.metric_by_name('smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed').as_double()
    dmuls = action.metric_by_name('smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed').as_double()
    dfmas = action.metric_by_name('smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed').as_double()
    fadds = action.metric_by_name('smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed').as_double()
    fmuls = action.metric_by_name('smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed').as_double()
    ffmas = action.metric_by_name('smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed').as_double()
    cycles = action.metric_by_name('gpc__cycles_elapsed.max').as_double()
    dadds = int(dadds * cycles)
    dmuls = int(dmuls * cycles)
    dfmas = int(dfmas * cycles)
    fadds = int(fadds * cycles)
    fmuls = int(fmuls * cycles)
    ffmas = int(ffmas * cycles)
    dW = dadds + dmuls + 2*dfmas
    fW = fadds + fmuls + 2*ffmas
    return {'dadds': dadds, 'dmuls': dmuls, 'dfmas': dfmas, 'fadds': fadds, 'fmuls': fmuls, 'ffmas': ffmas, 'dW': dW,
            'fW': fW}


def get_achieved_bytes(action: ncu_report.IAction) -> int:
    Q = action.metric_by_name('dram__bytes_write.sum').as_uint64()
    Q += action.metric_by_name('dram__bytes_read.sum').as_uint64()
    return Q


def get_achieved_performance(action: ncu_report.IAction) -> Tuple[float, float]:
    """
    Gets the achieved performance given the ncu action.

    :param action: The ncu action object
    :type action: ncu_report.IAction
    :return: The achieved performance and bandwidth in flop/s and byte/s
    :rtype: Tuple[float, float]
    """
    # total bytes written and read
    Q = float(get_achieved_bytes(action))
    # total double flop
    W = float(get_achieved_work(action)['dW'])
    # assume here that the unit is always nseconds
    runtime = action.metric_by_name('gpu__time_duration.sum').as_double() / 1e9
    return (W / runtime, Q / runtime)
