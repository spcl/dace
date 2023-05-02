from typing import Tuple, Dict, List
# import sys
# sys.path.insert(0, '/apps/ault/spack/opt/spack/linux-centos8-zen/gcc-8.4.1/cuda-11.8.0-fjdnxm6yggxxp75sb62xrxxmeg4s24ml/nsight-compute-2022.3.0/extras/python/')
from .ncu_report import IAction, load_report


def get_action(filename: str) -> IAction:
    """
    Gets the action inside the given ncu-rep file. Assumes there is only one range and action

    :param filename: The path/filename of the ncu-report
    :type filename: str
    """
    my_context = load_report(filename)
    num_ranges = my_context.num_ranges()
    if num_ranges > 1:
        print(f"WARNING: More than one range found in {filename} only taking the first")
    if num_ranges == 0:
        print(f"ERROR: There are no ranges in {filename}")
        return None
    my_range = my_context.range_by_idx(0)
    num_actions = my_range.num_actions()
    if num_actions > 1:
        print(f"WARNING: More than one action found in {filename} only taking the first")
    if num_actions == 0:
        print(f"ERROR: There are no actions in {filename}")
        return None
    my_action = my_range.action_by_idx(0)
    return my_action


def get_all_actions(filename: str) -> List[IAction]:
    """
    Gets all the actions inside the given ncu-rep file. Assumes there is only one range

    :param filename: The path/filename of the ncu-report
    :type filename: str
    """
    actions = []
    my_context = load_report(filename)
    num_ranges = my_context.num_ranges()
    if num_ranges > 1:
        print(f"WARNING: More than one range found in {filename} only taking the first")
    my_range = my_context.range_by_idx(0)
    num_actions = my_range.num_actions()
    for idx in range(num_actions):
        actions.append(my_range.action_by_idx(idx))
    return actions


def action_list_to_dict(actions: List[IAction]) -> Dict[str, List[IAction]]:
    """
    Converts the given list of actions into a dictionary. The keys are the action names, the values are a list of all
    actions with the same name

    :param actions: The list of actions
    :type actions: List[IAction]
    :return: The dictionary with the actions
    :rtype: Dict[str, List[IAction]]
    """
    action_dict = {}
    for action in actions:
        name = action.name()
        if name in action_dict:
            action_dict[name].append(action)
        else:
            action_dict[name] = [action]
    return action_dict


def get_peak_performance(action: IAction) -> Tuple[float, float]:
    """
    Gets the peak performance given the ncu action.

    :param action: The ncu action object
    :type action: IAction
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


def get_achieved_work(action: IAction) -> Dict[str, int]:
    """
    Returns the amount of work achieved as a dictionary containg the amount of double and single precision adds, muls,
    fmas and work (dW, fW)
    :param action: The ncu action object
    :type action: IAction
    :return: Dictionary with all data, keys are: dadds, dmuls, dfmas, fadds, fmuls, ffmas, dW, fW. Values are in flop
    :rtype: Dict[str, int]
    """
    dadds = action.metric_by_name('smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed').as_double()
    dmuls = action.metric_by_name('smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed').as_double()
    dfmas = action.metric_by_name('smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed').as_double()
    fadds = action.metric_by_name('smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed').as_double()
    fmuls = action.metric_by_name('smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed').as_double()
    ffmas = action.metric_by_name('smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed').as_double()
    cycles = float(get_cycles(action))
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


def get_achieved_bytes(action: IAction) -> int:
    Q = action.metric_by_name('dram__bytes_write.sum').as_uint64()
    Q += action.metric_by_name('dram__bytes_read.sum').as_uint64()
    return Q


def get_achieved_performance(action: IAction) -> Tuple[float, float]:
    """
    Gets the achieved performance given the ncu action.

    :param action: The ncu action object
    :type action: IAction
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


def get_runtime(action: IAction) -> float:
    """
    Returns the runtime in seconds

    :param action: The ncu actio object
    :type action: IAction
    :return: The runtime in seconds
    :rtype: float
    """
    return action.metric_by_name('gpu__time_duration.sum').as_double() / 1e9


def get_cycles(action: IAction) -> int:
    """
    Return the number of elapsed GPU cycles

    :param action: The ncu action object
    :type action: Action
    :return: The number of cycles
    :rtype: int
    """
    return action.metric_by_name('gpc__cycles_elapsed.max').as_uint64()
