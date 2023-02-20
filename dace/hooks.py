# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
Module that provides hooks that can be used to extend DaCe functionality.
"""

from typing import Any, Callable, Generator, List, Optional, Tuple, Union, ContextManager, TYPE_CHECKING
from contextlib import contextmanager, ExitStack
import pydoc
from dace import config

if TYPE_CHECKING:
    from dace.sdfg import SDFG
    from dace.codegen.compiled_sdfg import CompiledSDFG

CallHookType = Callable[['SDFG'], None]
CompiledCallHookType = Callable[['CompiledSDFG', Tuple[Any, ...]], None]
GeneratorType = Generator[Any, None, None]

# Global list of hooks
_SDFG_CALL_HOOKS: List[GeneratorType] = []
_COMPILED_SDFG_CALL_HOOKS: List[GeneratorType] = []


def register_sdfg_call_hook(hook: GeneratorType) -> int:
    """
    Registers a hook that is called when an SDFG is called.
    
    :param hook: The hook to register.
    :return: The unique identifier of the hook (for removal).
    """
    hook_id = len(_SDFG_CALL_HOOKS)
    _SDFG_CALL_HOOKS.append(hook)
    return hook_id


def register_compiled_sdfg_call_hook(hook: GeneratorType) -> int:
    """
    Registers a hook that is called when a compiled SDFG is called.
    
    :param hook: The hook to register.
    :return: The unique identifier of the hook (for removal).
    """
    hook_id = len(_COMPILED_SDFG_CALL_HOOKS)
    _COMPILED_SDFG_CALL_HOOKS.append(hook)
    return hook_id


def unregister_sdfg_call_hook(hook_id: int):
    """
    Unregisters an SDFG call hook.

    :param hook_id: The unique identifier of the hook.
    """
    _SDFG_CALL_HOOKS[hook_id] = None


def unregister_compiled_sdfg_call_hook(hook_id: int):
    """
    Unregisters a compiled SDFG call hook.
    
    :param hook_id: The unique identifier of the hook.
    """
    _COMPILED_SDFG_CALL_HOOKS[hook_id] = None


@contextmanager
def on_call(before_call: CallHookType, after_call: Optional[CallHookType] = None):
    """
    Context manager that registers a hook to be called before an SDFG is
    compiled and run.

    Example use:

    .. code-block:: python

        with dace.on_call(lambda sdfg: print('Before', sdfg.name),
                          lambda sdfg: print('After', sdfg.name)):
            program(...)
            # ...


    :param hook: A function to be called before the SDFG call or a context
                 manager to be called around it.
    :param after_call: If a function was given to ``hook``, an optional
                       function to be called after the SDFG call.
    """
    hook = _as_context_manager(before_call, after_call)
    hook_id = register_sdfg_call_hook(hook)
    try:
        yield
    finally:
        unregister_sdfg_call_hook(hook_id)


@contextmanager
def on_compiled_sdfg_call(before_call: CompiledCallHookType, after_call: Optional[CompiledCallHookType] = None):
    """
    Context manager that registers a hook to be called before a compiled SDFG is
    invoked.

    :param hook: A function to be called before the SDFG call or a context
                 manager to be called around it.
    :param after_call: If a function was given to ``hook``, an optional
                       function to be called after the SDFG call.
    """
    hook = _as_context_manager(before_call, after_call)
    hook_id = register_compiled_sdfg_call_hook(hook)
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


class SDFGProfiler:
    """
    Simple profiler class that invokes an SDFG multiple times and prints/stores
    the wall-clock time.
    """
    def __init__(self, repetitions: int) -> None:
        if repetitions < 1:
            raise ValueError('Number of repetitions must be at least 1')
        self.reps = repetitions
        self.csdfg = None
        self.orig_func = None

    def __call__(self, csdfg: 'CompiledSDFG', args: Tuple[Any, ...]):
        from dace.frontend.operations import timethis
        timethis(csdfg._sdfg, 'DaCe', 0, csdfg._cfunc, csdfg._libhandle, *args, REPS=self.reps)

        # Ensure internal SDFG will not be called by triggering an exception
        self.csdfg = csdfg
        self.orig_func = csdfg._cfunc
        csdfg._cfunc = None

        return self

    def __enter__(self):
        # Do nothing
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore state and return True regardless of the raised exception
        self.csdfg._cfunc = self.orig_func
        self.csdfg = self.orig_func = None
        return True


@contextmanager
def profile(repetitions: int = 100):
    """
    Context manager that enables profiling of an entire SDFG with wall-clock timing.
    
    :param repetitions: Number of repetitions to call the SDFG.
    """
    profiler = SDFGProfiler(repetitions)
    with on_compiled_sdfg_call(profiler):
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
        register_hook_func(_as_context_manager(hookfunc))


def _install_hooks_from_config():
    _install_hooks_helper('call_hooks', register_sdfg_call_hook)
    _install_hooks_helper('compiled_sdfg_call_hooks', register_sdfg_call_hook)

    # Convenience hooks
    if config.Config.get_bool('profiling'):
        repetitions = config.Config.get('treps')
        register_compiled_sdfg_call_hook(SDFGProfiler(repetitions))


_install_hooks_from_config()
