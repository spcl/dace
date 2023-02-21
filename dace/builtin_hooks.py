# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
A set of built-in hooks.
"""
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dace.sdfg import SDFG

@contextmanager
def profile(repetitions: int = 100, warmup: int = 0):
    """
    Context manager that enables profiling of each called DaCe program. If repetitions is greater than 1, the
    program is run multiple times and the average execution time is reported.

    Example usage:

    .. code-block:: python

        with dace.profile(repetitions=100) as profiler:
            some_program(...)
            # ...
            other_program(...)

        # Print all execution times of the last called program (other_program)
        print(profiler.times[-1])


    :param repetitions: The number of times to run each DaCe program.
    :param warmup: Number of additional repetitions to run the program without measuring time.
    :note: Running functions multiple times may affect the results of the program.
    """
    from dace.frontend.operations import CompiledSDFGProfiler  # Avoid circular import
    from dace.hooks import on_compiled_sdfg_call, _COMPILED_SDFG_CALL_HOOKS

    # If already profiling, return existing profiler and change its properties
    for hook in _COMPILED_SDFG_CALL_HOOKS:
        if isinstance(hook, CompiledSDFGProfiler):
            hook.times.clear()
            hook.repetitions = repetitions
            hook.warmup = warmup
            yield hook
            return

    profiler = CompiledSDFGProfiler(repetitions, warmup)

    with on_compiled_sdfg_call(context_manager=profiler):
        yield profiler


def cli_optimize_on_call(sdfg: 'SDFG'):
    """
    Calls a command-line interface for interactive SDFG transformations
    on every DaCe program call.

    :param sdfg: The current SDFG to optimize.
    """

    from dace.transformation.optimizer import SDFGOptimizer
    opt = SDFGOptimizer(sdfg)
    return opt.optimize()
