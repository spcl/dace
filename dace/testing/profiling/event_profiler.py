"""Utilities for profiling PyTorch."""

import statistics

import torch
import torch.cuda
import tabulate

# Validate that we have CUDA and initialize it on load.
from dace.testing.profiling import binary_utils

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise RuntimeError('No CUDA support or GPUs available')
torch.cuda.init()

# A device for CUDA.
cuda_device = torch.device('cuda:0')

_DEFAULT_WAIT_TIME = 0.001  # Default wait kernel time, 1 ms.


class CudaTimer:
    """ Time CUDA kernels with CUDA events.

        :param n:
    """
    def __init__(self, n: int):
        self.events = []
        for _ in range(n + 1):
            self.events.append(torch.cuda.Event(enable_timing=True))
        self.cur = 0

    def start(self):
        """ Start timing the first region. """
        self.events[0].record()
        self.cur = 1

    def next(self):
        """ Start timing for the next region. """
        if self.cur >= len(self.events):
            raise RuntimeError('Trying to time too many regions')
        self.events[self.cur].record()
        self.cur += 1

    def end(self):
        """ Finish timing. """
        if self.cur != len(self.events):
            raise RuntimeError('Called end too early')
        self.events[-1].synchronize()

    def get_times(self):
        """ Return measured time for each region. """
        times = []
        for i in range(1, len(self.events)):
            times.append(self.events[i - 1].elapsed_time(self.events[i]))
        return times


def time_one(funcs, launch_wait=True, timer=None, name='', func_names=None):
    """ Run and time a single iteration of funcs.
        funcs is a list functions that do GPU work.

        :param launch_wait: if True, launches a wait kernel before measuring to hide kernel launch latency.
        :param timer: an already existing instance of CudaTimer. If ``None``, one will be created.
        :param name: a name to be used for NVTX ranges.
        :param func_names: a list of names to be used for NVTX ranges for each function.
        :return: the time, in ms, of each function in funcs.
    """
    if timer is None:
        timer = CudaTimer(len(funcs))
    if launch_wait:
        binary_utils.gpu_wait(_DEFAULT_WAIT_TIME)
    torch.cuda.nvtx.range_push(name + ' Iter')
    timer.start()
    if not func_names:
        func_names = ['func ' + str(i) for i in range(len(funcs))]
    for f, name in zip(funcs, func_names):
        torch.cuda.nvtx.range_push(name)
        f()
        torch.cuda.nvtx.range_pop()
        timer.next()
    timer.end()
    torch.cuda.nvtx.range_pop()
    return timer.get_times()


def time_funcs(funcs,
               name='',
               func_names=None,
               num_iters=100,
               warmups=5,
               launch_wait=False):
    """ Run and time funcs.

        :param funcs: a list of functions that do GPU work.
        :param name: a name to be used for NVTX ranges.
        :param func_names: a list of names to be used for NVTX ranges for each function.
        :param num_iters: the number of iterations to perform.
        :param warmups:  the number of warmup iterations to perform.
        :param launch_wait: if True, launches a wait kernel before measuring to hide kernel launch latency.
        :return: the time, in ms, of each function in funcs on each iteration.
    """
    timer = CudaTimer(len(funcs))
    binary_utils.start_cuda_profiling()
    torch.cuda.nvtx.range_push(name + ' Warmup')
    for _ in range(warmups):
        for f in funcs:
            f()
    torch.cuda.nvtx.range_pop()
    times = [list() for _ in range(len(funcs))]
    for _ in range(num_iters):
        iter_times = time_one(funcs,
                              launch_wait=launch_wait,
                              timer=timer,
                              name=name,
                              func_names=func_names)
        for i, t in enumerate(iter_times):
            times[i].append(t)
    binary_utils.stop_cuda_profiling()
    return times


def print_time_statistics(times, func_names):
    """ Print timing statistics.
    :param times: the result of time_funcs.
    :param func_names: a name to use for each function timed.
    """
    headers = ['Name', 'Min', 'Mean', 'Median', 'Stdev', 'Max']
    rows = []
    for name, func_time in zip(func_names, times):
        rows.append([
            name,
            min(func_time),
            statistics.mean(func_time),
            statistics.median(func_time),
            statistics.stdev(func_time) if len(func_time) > 1 else 0.0,
            max(func_time)
        ])
    print(
        tabulate.tabulate(rows,
                          headers=headers,
                          floatfmt='.4f',
                          tablefmt='github'))
