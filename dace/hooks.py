# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
Module that provides hooks that can be used to extend DaCe functionality.
"""

from typing import Any, Callable, Generator, List, Optional, Tuple, Union, ContextManager, TYPE_CHECKING
from contextlib import contextmanager, ExitStack
import pydoc
from dace import config
import warnings

if TYPE_CHECKING:
    from dace.sdfg import SDFG
    from dace.codegen.compiled_sdfg import CompiledSDFG

CallHookType = Callable[['SDFG'], None]
CompiledCallHookType = Callable[['CompiledSDFG', Tuple[Any, ...]], None]
GeneratorType = Generator[Any, None, None]

# Global list of hooks
_SDFG_CALL_HOOKS: List[GeneratorType] = []
_COMPILED_SDFG_CALL_HOOKS: List[GeneratorType] = []


def _register_hook(hook_list: List[GeneratorType], before_hook: Optional[Callable[..., None]],
                   after_hook: Optional[Callable[..., None]], context_manager: Optional[GeneratorType]) -> int:
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
        def hook(*args, **kwargs):
            if before_hook is not None:
                before_hook(*args, **kwargs)
            try:
                yield
            finally:
                if after_hook is not None:
                    after_hook(*args, **kwargs)

    hook_id = len(hook_list)
    hook_list.append(hook)
    return hook_id


def register_sdfg_call_hook(*,
                            before_hook: Optional[CallHookType] = None,
                            after_hook: Optional[CallHookType] = None,
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
                                     before_hook: Optional[CompiledCallHookType] = None,
                                     after_hook: Optional[CompiledCallHookType] = None,
                                     context_manager: Optional[GeneratorType] = None) -> int:
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
def on_call(*,
            before: Optional[CallHookType] = None,
            after: Optional[CallHookType] = None,
            context_manager: Optional[GeneratorType] = None):
    """
    Context manager that registers a function to be called around each SDFG call.
    Use this to modify the SDFG before it is compiled and run.

    For example, to print the SDFG before it is compiled and run:

    .. code-block:: python

        # Will print "some_program was called"
        with dace.hooks.on_call(before=lambda sdfg: print(f'{sdfg.name} was called')):
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
def on_compiled_sdfg_call(*,
                          before: Optional[CompiledCallHookType] = None,
                          after: Optional[CompiledCallHookType] = None,
                          context_manager: Optional[GeneratorType] = None):
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
        
        with dace.hooks.on_compiled_sdfg_call(context_manager=time_compiled_sdfg):
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
# Input type


def _as_context_manager(begin_func: Union[Callable[..., Any], ContextManager],
                        end_func: Optional[Callable[..., Any]] = None) -> GeneratorType:
    """
    Returns a context manager from a begin and end functions, if not already given.

    :param begin_func: A callable object or context manager.
    :param end_func: An optional callable object.
    :return: Context manager that calls the given functions.
    """
    # Already a context manager
    if hasattr(begin_func, '__enter__'):
        if end_func is not None:
            raise ValueError('A context manager cannot be given with an end function')
        return begin_func

    if end_func is None:

        @contextmanager
        def begin_ctxmgr(*args, **kwargs):
            begin_func(*args, **kwargs)
            yield

        return begin_ctxmgr

    @contextmanager
    def begin_end_ctxmgr(*args, **kwargs):
        begin_func(*args, **kwargs)
        yield
        end_func(*args, **kwargs)

    return begin_end_ctxmgr


##############################################################################
# Invocation


@contextmanager
def invoke_sdfg_call_hooks(sdfg: 'SDFG'):
    """
    Internal context manager that calls all SDFG call hooks in their registered order.
    """
    if not _SDFG_CALL_HOOKS:
        yield sdfg
        return
    with ExitStack() as stack:
        for hook in _SDFG_CALL_HOOKS:
            if hook is None:
                continue
            new_sdfg = stack.enter_context(hook(sdfg))
            if new_sdfg is not None:
                sdfg = new_sdfg
        yield sdfg


@contextmanager
def invoke_compiled_sdfg_call_hooks(compiled_sdfg: 'CompiledSDFG', args: Tuple[Any, ...]):
    """
    Internal context manager that calls all compiled SDFG call hooks in their registered order.
    """
    if not _COMPILED_SDFG_CALL_HOOKS:
        yield compiled_sdfg
        return
    with ExitStack() as stack:
        for hook in _COMPILED_SDFG_CALL_HOOKS:
            if hook is None:
                continue
            new_compiled_sdfg = stack.enter_context(hook(compiled_sdfg, args))
            if new_compiled_sdfg is not None:
                compiled_sdfg = new_compiled_sdfg

        yield compiled_sdfg


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
    from dace.frontend.operations import CompiledSDFGProfiler  # Avoid circular import

    # If already profiling, return existing profiler and change its properties
    for hook in _COMPILED_SDFG_CALL_HOOKS:
        if isinstance(hook, CompiledSDFGProfiler):
            hook.times.clear()
            hook.repetitions = repetitions
            yield hook
            return

    profiler = CompiledSDFGProfiler(repetitions)

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


##########################################################################
# Install hooks from configuration upon import


def _install_hooks_helper(config_name: str, register_hook_func: Callable[[GeneratorType], None]):
    hooklist = config.Config.get(config_name)
    if not hooklist:
        return
    hooklist = hooklist.split(',')
    for hook in hooklist:
        hookfunc = pydoc.locate(hook)
        if hookfunc is None:
            warnings.warn(f'Hook "{hook}" was not found or could not be loaded. Skipping.')
            continue
        register_hook_func(context_manager=_as_context_manager(hookfunc))


def _install_hooks_from_config():
    _install_hooks_helper('call_hooks', register_sdfg_call_hook)
    _install_hooks_helper('compiled_sdfg_call_hooks', register_sdfg_call_hook)

    # Convenience hooks
    if config.Config.get_bool('profiling'):
        from dace.frontend.operations import CompiledSDFGProfiler
        register_compiled_sdfg_call_hook(context_manager=CompiledSDFGProfiler())


_install_hooks_from_config()
