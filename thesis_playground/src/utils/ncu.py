from typing import Tuple, Dict, List
from .ncu_report import IAction, load_report
from math import isnan
import re


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
        print(f"WARNING: More than one action found in {filename} only taking the first {my_range.action_by_idx(0)}"
              f"of {num_actions}")
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


def get_all_actions_filtered(filename: str, ignore_re: str) -> List[IAction]:
    """
    Gets all the actions inside the given ncu-rep file which do not match the given regex.
    Assumes there is only one range

    :param filename: The path/filename of the ncu-report
    :type filename: str
    :param ignore_re: The regex of any action names to ignore
    :type ignore_re: str
    :return: List of actions
    :rtype: List[IAction]
    """
    actions = [a for a in get_all_actions(filename) if not re.match(ignore_re, a.name())]
    return actions


def get_all_actions_matching_re(filename: str, re_str: str) -> List[IAction]:
    """
    Gets all the actions inside the given ncu-rep file which match the given regex.
    Assumes there is only one range

    :param filename: The path/filename of the ncu-report
    :type filename: str
    :param re_str: The regex all action names need to match against
    :type re_str: str
    :return: List of actions
    :rtype: List[IAction]
    """
    actions = [a for a in get_all_actions(filename) if re.match(re_str, a.name())]
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
    dadds = int(dadds * cycles) if not isnan(dadds) and not isnan(cycles) else None
    dmuls = int(dmuls * cycles) if not isnan(dmuls) and not isnan(cycles) else None
    dfmas = int(dfmas * cycles) if not isnan(dfmas) and not isnan(cycles) else None
    fadds = int(fadds * cycles) if not isnan(fadds) and not isnan(cycles) else None
    fmuls = int(fmuls * cycles) if not isnan(fmuls) and not isnan(cycles) else None
    ffmas = int(ffmas * cycles) if not isnan(ffmas) and not isnan(cycles) else None
    dW = dadds + dmuls + 2*dfmas if dadds is not None and dmuls is not None and dfmas is not None else None
    fW = fadds + fmuls + 2*ffmas if fadds is not None and fmuls is not None and ffmas is not None else None
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
    Q = get_achieved_bytes(action)
    Q = float(Q) if Q is not None else Q
    # total double flop
    W = get_achieved_work(action)['dW']
    W = float(W) if W is not None else W
    # assume here that the unit is always nseconds
    runtime = action.metric_by_name('gpu__time_duration.sum').as_double() / 1e9
    performance = W / runtime if W is not None else None
    memory = Q / runtime if Q is not None else None
    return (performance, memory)


def get_runtime(action: IAction) -> float:
    """
    Returns the runtime in seconds

    :param action: The ncu actio object
    :type action: IAction
    :return: The runtime in seconds
    :rtype: float
    """
    print(f"[get_runtime] {type(action)}")
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

