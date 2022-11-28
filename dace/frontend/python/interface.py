# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Python interface for DaCe functions. """

import inspect
from functools import wraps
from typing import Any, Callable, Deque, Dict, Generator, Optional, Tuple, TypeVar, Union, overload

from dace import dtypes
from dace.dtypes import paramdec
from dace.frontend.python import ndloop, parser, tasklet_runner

#############################################

T = TypeVar('T')

# Type hints specifically written for the @dace.program decorator
F = TypeVar('F', bound=Callable[..., Any])


@overload
def program(f: F) -> parser.DaceProgram:
    ...


@overload
def program(*args,
            auto_optimize=False,
            device=dtypes.DeviceType.CPU,
            recreate_sdfg: bool = True,
            regenerate_code: bool = True,
            recompile: bool = True,
            constant_functions=False,
            **kwargs) -> Callable[..., parser.DaceProgram]:
    ...


@paramdec
def program(f: F,
            *args,
            auto_optimize=False,
            device=dtypes.DeviceType.CPU,
            recreate_sdfg: bool = True,
            regenerate_code: bool = True,
            recompile: bool = True,
            constant_functions=False,
            **kwargs) -> Callable[..., parser.DaceProgram]:
    """
    Entry point to a data-centric program. For methods and ``classmethod``s, use
    ``@dace.method``.

    :param f: The function to define as the entry point.
    :param auto_optimize: If True, applies automatic optimization heuristics
                          on the generated DaCe program during compilation.
    :param device: Transform the function to run on the target device.
    :param recreate_sdfg: Whether to recreate the SDFG from the Python code. If False, the SDFG will be loaded from the
                          cache (``<build folder>/<program name>/program.sdfg``) if it exists.
                          Use this if you want to modify the SDFG after the first call to the function.
    :param regenerate_code: Whether to regenerate the code from the SDFG. If False, the code in the build folder will be
                            used if it exists. Use this if you want to modify the generated code without DaCe overriding
                            it.
    :param recompile: Whether to recompile the code. If False, the library in the build folder will be used if it exists,
                      without recompiling it.
    :param constant_functions: If True, assumes all external functions that do
                               not depend on internal variables are constant.
                               This will hardcode their return values into the
                               resulting program.
    :note: If arguments are defined with type hints, the program can be compiled
           ahead-of-time with ``.compile()``.
    """

    # Parses a python @dace.program function and returns an object that can
    # be translated
    return parser.DaceProgram(f,
                              args,
                              kwargs,
                              auto_optimize,
                              device,
                              constant_functions,
                              recreate_sdfg=recreate_sdfg,
                              regenerate_code=regenerate_code,
                              recompile=recompile)


function = program


@overload
def method(f: F) -> parser.DaceProgram:
    ...


@overload
def method(*args,
           auto_optimize=False,
           device=dtypes.DeviceType.CPU,
           constant_functions=False,
           **kwargs) -> parser.DaceProgram:
    ...


@paramdec
def method(f: F,
           *args,
           auto_optimize=False,
           device=dtypes.DeviceType.CPU,
           recreate_sdfg: bool = True,
           regenerate_code: bool = True,
           recompile: bool = True,
           constant_functions=False,
           **kwargs) -> parser.DaceProgram:
    """ 
    Entry point to a data-centric program that is a method or  a ``classmethod``. 

    :param f: The method to define as the entry point.
    :param auto_optimize: If True, applies automatic optimization heuristics
                          on the generated DaCe program during compilation.
    :param device: Transform the function to run on the target device.
    :param recreate_sdfg: Whether to recreate the SDFG from the Python code. If False, the SDFG will be loaded from the
                          cache (``<build folder>/<program name>/program.sdfg``) if it exists.
                          Use this if you want to modify the SDFG after the first call to the function.
    :param regenerate_code: Whether to regenerate the code from the SDFG. If False, the code in the build folder will be
                            used if it exists. Use this if you want to modify the generated code without DaCe overriding
                            it.
    :param recompile: Whether to recompile the code. If False, the library in the build folder will be used if it exists,
                      without recompiling it.
    :param constant_functions: If True, assumes all external functions that do
                               not depend on internal variables are constant.
                               This will hardcode their return values into the
                               resulting program.
    :note: If arguments are defined with type hints, the program can be compiled
           ahead-of-time with ``.compile()``.    
    """

    # Create a wrapper class that can bind to the object instance
    class MethodWrapper:

        def __init__(self):
            self.wrapped: Dict[int, parser.DaceProgram] = {}

        def __get__(self, obj, objtype=None) -> parser.DaceProgram:
            # Modify wrapped instance as necessary, only clearing
            # compiled program cache if needed.
            objid = id(obj)
            if objid in self.wrapped:
                return self.wrapped[objid]
            prog = parser.DaceProgram(f,
                                      args,
                                      kwargs,
                                      auto_optimize,
                                      device,
                                      constant_functions,
                                      recreate_sdfg=recreate_sdfg,
                                      regenerate_code=regenerate_code,
                                      recompile=recompile,
                                      method=True)
            prog.methodobj = obj
            self.wrapped[objid] = prog
            return prog

        def __call__(self, *call_args, **call_kwargs):
            # When called as-is, treat as an unbounded function (dace.program)
            if None not in self.wrapped:
                prog = parser.DaceProgram(f,
                                          args,
                                          kwargs,
                                          auto_optimize,
                                          device,
                                          constant_functions,
                                          recreate_sdfg=recreate_sdfg,
                                          regenerate_code=regenerate_code,
                                          recompile=recompile,
                                          method=False)
                self.wrapped[None] = prog
            else:
                prog = self.wrapped[None]

            return prog(*call_args, **call_kwargs)

    return MethodWrapper()


# DaCe functions


# Dataflow constructs
class MapGenerator:
    """
    An SDFG map generator class that allows applying operators on it, used
    for syntactic sugar.
    """

    def __init__(self, rng: Union[slice, Tuple[slice]]):
        self.rng = rng

    def __matmul__(self, schedule: dtypes.ScheduleType):
        """
        Syntactic sugar for specifying the schedule of the map.
        This enables controlling the hardware scheduling of the map as follows:

        .. code-block:: python

            for i, j in dace.map[0:N, 0:M] @ ScheduleType.GPU_Global:
                b[i, j] = a[i, j] + 1
        """
        # Ignored in Python mode
        return self

    def __iter__(self):
        return ndloop.ndrange(self.rng)


class MapMetaclass(type):
    """ Metaclass for map, to enable ``dace.map[0:N]`` syntax. """

    @classmethod
    def __getitem__(cls, rng: Union[slice, Tuple[slice]]) -> Generator[Tuple[int], None, None]:
        """ 
        Iterates over an N-dimensional region in parallel.

        :param rng: A slice or a tuple of multiple slices, representing the
                    N-dimensional range to iterate over.
        :return: Generator of N-dimensional tuples of iterates.
        """
        return MapGenerator(rng)


class map(metaclass=MapMetaclass):
    """ A Map is representation of parallel execution, containing
        an integer set (Python range) for which its contents are run 
        concurrently. Written in Python as a loop with the following
        syntax: `for i, j in dace.map[1:20, 0:N]:`.
    """
    pass


class consume:

    def __init__(self, stream: Deque[T], processing_elements: int = 1, condition: Optional[Callable[[], bool]] = None):
        """ 
        Consume is a scope, like ``Map``, that creates parallel execution.
        Unlike `Map`, it creates a producer-consumer relationship between an
        input stream and the contents. The contents are run by the given number
        of processing elements, who will try to pop elements from the input
        stream until a given quiescence condition is reached. 

        :param stream: The stream to pop from.
        :param processing_elements: The number of processing elements to use.
        :param condition: A custom condition for stopping to consume. If None,
                          loops until ``stream`` is empty.
        """
        self.stream = stream
        self.pes = processing_elements  # Ignored in Python mode
        self.condition = condition or (lambda: len(stream) > 0)

    def __iter__(self) -> Generator[T, None, None]:
        """
        Consume stream (pop elements) until condition is met, or until the
        stream is empty, if no condition is given.
        """
        while self.condition():
            yield self.stream.pop()


class TaskletMetaclass(type):
    """ Metaclass for tasklet, to enable ``with dace.tasklet:`` syntax. """

    def __enter__(self):
        # Parse and run tasklet
        frame = inspect.stack()[1][0]
        filename = inspect.getframeinfo(frame, context=0).filename
        tasklet_ast = tasklet_runner.get_tasklet_ast(frame=frame)
        tasklet_runner.run_tasklet(tasklet_ast, filename, frame.f_globals, frame.f_locals)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Tasklets always raise exceptions (NameError due to the memlet
        # syntax and undefined connector names, or TypeError due to bad shifts).
        # Thus, their contents are skipped.
        return True


class tasklet(metaclass=TaskletMetaclass):
    """ 
    A general procedure that cannot access any memory apart from incoming
    and outgoing memlets. Memlets use the shift operator, an example of
    a tasklet is::

        with dace.tasklet:
            a << A[i, j]  # Memlet going from A to a
            b = a + 5
            b >> B[i, j]  # Memlet going out of the tasklet to B


    The DaCe framework cannot analyze these tasklets for optimization. 
    """

    def __init__(self, language: Union[str, dtypes.Language] = dtypes.Language.Python):
        if isinstance(language, str):
            language = dtypes.Language[language]
        self.language = language

    def __enter__(self):
        if self.language != dtypes.Language.Python:
            raise NotImplementedError('Cannot run non-Python tasklet in Python')

        # Parse and run tasklet
        frame = inspect.stack()[1][0]
        filename = inspect.getframeinfo(frame, context=0).filename
        tasklet_ast = tasklet_runner.get_tasklet_ast(frame=frame)
        tasklet_runner.run_tasklet(tasklet_ast, filename, frame.f_globals, frame.f_locals)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Tasklets always raise exceptions (NameError due to the memlet
        # syntax and undefined connector names, or TypeError due to bad shifts).
        # Thus, their contents are skipped.
        return True


def unroll(generator):
    """
    Explicitly annotates that a loop should be unrolled during parsing.

    :param generator: The original generator to loop over.
    :note: Only use with stateless and compile-time evaluateable loops!
    """
    yield from generator


def nounroll(generator):
    """
    Explicitly annotates that a loop should not be unrolled during parsing.

    :param generator: The original generator to loop over.
    """
    yield from generator


def in_program() -> bool:
    """
    Returns True if in a DaCe program parsing context. This function can be used to test whether the current
    code runs inside the ``@dace.program`` parser.
    
    :return: True if in a DaCe program parsing context, or False otherwise.
    """
    return False
