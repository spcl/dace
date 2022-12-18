# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
Module that provides hooks that can be used to extend DaCe functionality.
"""

from typing import Any, Callable, Generator, List, Optional, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    from dace.sdfg import SDFG

HookType = Callable[['SDFG'], Any]
GeneratorType = Callable[['SDFG'], Generator[Any, None, None]]
CHookType = Callable[..., Any]
CGeneratorType = Callable[..., Generator[Any, None, None]]

# Global list of hooks
_SDFG_CALL_HOOKS: List[GeneratorType] = []
_COMPILED_SDFG_CALL_HOOKS: List[GeneratorType] = []


def _register_hook(hook_list: List[GeneratorType], before_hook: Optional[HookType], after_hook: Optional[HookType],
                   context_manager: Optional[GeneratorType]) -> int:
    """
    Internal function that registers function or context manager hooks to be called.

    :param hook_list: The list of hooks to register to.
    :param before_hook: An optional hook to call before the event.
    :param after_hook: An optional hook to call after the event.
    :param context_manager: A context manager to use around the event. This field
                            can only be used if both ``before_hook`` and ``after_hook`` are ``None``.
    :return: The unique identifier of the hook (for removal).
    """
    if before_hook is None and after_hook is None and context_manager is None:
        raise ValueError('At least one of before_hook, after_hook, or context_manager must be specified')
    if (before_hook is not None or after_hook is not None) and context_manager is not None:
        raise ValueError('Cannot specify both before_hook/after_hook and context_manager')

    if context_manager is not None:
        hook = context_manager
    else:
        # Wrap the hooks in a context manager
        @contextmanager
        def hook(sdfg: 'SDFG'):
            if before_hook is not None:
                before_hook(sdfg)
            try:
                yield
            finally:
                if after_hook is not None:
                    after_hook(sdfg)

    hook_id = len(hook_list)
    hook_list.append(hook)
    return hook_id


def register_sdfg_call_hook(*,
                            before_hook: Optional[HookType] = None,
                            after_hook: Optional[HookType] = None,
                            context_manager: Optional[GeneratorType] = None) -> int:
    """
    Registers a hook that is called when an SDFG is called.
    
    :param before_hook: An optional hook to call before the SDFG is compiled and run.
    :param after_hook: An optional hook to call after the SDFG is compiled and run.
    :param context_manager: A context manager to use around the SDFG's compilation and running. This field
                            can only be used if both ``before_hook`` and ``after_hook`` are ``None``.
    :return: The unique identifier of the hook (for removal).
    """
    return _register_hook(_SDFG_CALL_HOOKS, before_hook, after_hook, context_manager)


def register_compiled_sdfg_call_hook(*,
                                     before_hook: Optional[HookType] = None,
                                     after_hook: Optional[CHookType] = None,
                                     context_manager: Optional[CGeneratorType] = None) -> int:
    """
    Registers a hook that is called when a compiled SDFG is called.
    
    :param before_hook: An optional hook to call before the compiled SDFG is called.
    :param after_hook: An optional hook to call after the compiled SDFG is called.
    :param context_manager: A context manager to use around the compiled SDFG's C function. This field
                            can only be used if both ``before_hook`` and ``after_hook`` are ``None``.
    :return: The unique identifier of the hook (for removal).
    """
    return _register_hook(_COMPILED_SDFG_CALL_HOOKS, before_hook, after_hook, context_manager)


def unregister_sdfg_call_hook(hook_id: int):
    """
    Unregisters an SDFG call hook.

    :param hook_id: The unique identifier of the hook.
    """
    if hook_id >= len(_SDFG_CALL_HOOKS):
        raise ValueError('Invalid hook ID')
    _SDFG_CALL_HOOKS[hook_id] = None


def unregister_compiled_sdfg_call_hook(hook_id: int):
    """
    Unregisters a compiled SDFG call hook.
    
    :param hook_id: The unique identifier of the hook.
    """
    if hook_id >= len(_COMPILED_SDFG_CALL_HOOKS):
        raise ValueError('Invalid hook ID')
    _COMPILED_SDFG_CALL_HOOKS[hook_id] = None


@contextmanager
def on_each_sdfg(*,
                 before: Optional[HookType] = None,
                 after: Optional[HookType] = None,
                 context_manager: Optional[GeneratorType] = None):
    """
    Context manager that registers a function to be called around each SDFG call.
    Use this to modify the SDFG before it is compiled and run.

    For example, to print the SDFG before it is compiled and run:

    .. code-block:: python

        # Will print "some_program was called"
        with dace.hooks.on_each_sdfg(before=lambda sdfg: print(f'{sdfg.name} was called')):
            some_program(...)

        # Alternatively, using a context manager
        @contextmanager
        def print_sdfg_name(sdfg: dace.SDFG):
            print(f'{sdfg.name} is going to be compiled and run')
            yield
            print(f'{sdfg.name} has finished running')
        
        with dace.hooks.on_each_sdfg(context_manager=print_sdfg_name):
            some_program(...)
        

    :param before: An optional function that is called before the SDFG is compiled and run. This function
                   should take an SDFG as its only argument.
    :param after: An optional function that is called after the SDFG is compiled and run. This function
                  should take an SDFG as its only argument.
    :param context_manager: A context manager to use around the SDFG's compilation and running. This field
                            can only be used if both ``before`` and ``after`` are ``None``.
    """
    hook_id = register_sdfg_call_hook(before_hook=before, after_hook=after, context_manager=context_manager)
    try:
        yield
    finally:
        unregister_sdfg_call_hook(hook_id)


@contextmanager
def on_each_compiled_sdfg(*,
                          before: Optional[CHookType] = None,
                          after: Optional[CHookType] = None,
                          context_manager: Optional[CGeneratorType] = None):
    """
    Context manager that registers a function to be called around each compiled SDFG call.
    Use this to wrap the compiled SDFG's C function call.

    For example, to time the execution of the compiled SDFG:

    .. code-block:: python

        @contextmanager
        def time_compiled_sdfg(csdfg: dace.codegen.compiled_sdfg.CompiledSDFG, *args, **kwargs):
            start = time.time()
            yield
            end = time.time()
            print(f'Compiled SDFG {csdfg.sdfg.name} took {end - start} seconds')
        
        with dace.hooks.on_each_compiled_sdfg(context_manager=time_compiled_sdfg):
            some_program(...)
            other_program(...)
        
    :param before: An optional function that is called before the compiled SDFG is called. This function
                   should take a compiled SDFG object, its arguments and keyword arguments.
    :param after: An optional function that is called after the compiled SDFG is called. This function
                  should take a compiled SDFG object, its arguments and keyword arguments.
    :param context_manager: A context manager to use around the compiled SDFG's C function. This field
                            can only be used if both ``before`` and ``after`` are ``None``.
    """
    hook_id = register_compiled_sdfg_call_hook(before_hook=before, after_hook=after, context_manager=context_manager)
    try:
        yield
    finally:
        unregister_compiled_sdfg_call_hook(hook_id)


##############################################################################
# Built-in hooks


@contextmanager
def profile(repetitions: int = 100):
    """
    Context manager that enables profiling of each called DaCe program. If repetitions is greater than 1, the
    program is run multiple times and the average execution time is reported.

    Example usage:

    .. code-block:: python

        with dace.profile(repetitions=100) as profiler:
            some_program(...)
            # ...
            other_program(...)

        # Print all execution times of the last called program
        print(profiler.times[-1])


    :param repetitions: The number of times to run each DaCe program.
    :note: Running functions multiple times may affect the results of the program.
    """
    if repetitions < 1:
        raise ValueError('Number of repetitions must be at least 1')
    from dace.frontend.operations import CompiledSDFGProfiler  # Avoid circular import

    profiler = CompiledSDFGProfiler(repetitions)

    with on_each_compiled_sdfg(context_manager=profiler.time_compiled_sdfg):
        yield profiler
