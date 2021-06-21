# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Python interface for DaCe functions. """

from networkx.algorithms.components.strongly_connected import condensation
from dace import dtypes
from dace.dtypes import paramdec
from dace.frontend.python import parser, ndloop, wrappers
from typing import (Any, Callable, Deque, Generator, Optional, Tuple, TypeVar,
                    overload, Union)

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
            **kwargs) -> parser.DaceProgram:
    ...


@paramdec
def program(f: F,
            *args,
            auto_optimize=False,
            device=dtypes.DeviceType.CPU,
            **kwargs) -> parser.DaceProgram:
    """ DaCe program, entry point to a data-centric program. """

    # Parses a python @dace.program function and returns an object that can
    # be translated
    return parser.DaceProgram(f, args, kwargs, auto_optimize, device)


function = program

# DaCe functions


# Dataflow constructs
class MapMetaclass(type):
    """ Metaclass for map, to enable ``dace.map[0:N]`` syntax. """
    @classmethod
    def __getitem__(
            cls, rng: Union[slice,
                            Tuple[slice]]) -> Generator[Tuple[int], None, None]:
        """ 
        Iterates over an N-dimensional region in parallel.
        :param rng: A slice or a tuple of multiple slices, representing the
                    N-dimensional range to iterate over.
        :return: Generator of N-dimensional tuples of iterates.
        """
        yield from ndloop.ndrange(rng)


class map(metaclass=MapMetaclass):
    """ A Map is representation of parallel execution, containing
        an integer set (Python range) for which its contents are run 
        concurrently. Written in Python as a loop with the following
        syntax: `for i, j in dace.map[1:20, 0:N]:`.
    """
    pass


class consume:
    def __init__(self,
                 stream: Deque[T],
                 processing_elements: int = 1,
                 condition: Optional[Callable[[], bool]] = None):
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
    @classmethod
    def __enter__(cls):
        # TODO: parse and run tasklet
        # TODO: reinstate tasklet simulator
        pass

    @classmethod
    def __exit__(cls, exc_type, exc_val, exc_tb):
        pass


class tasklet(metaclass=TaskletMetaclass):
    """ 
    A general procedure that cannot access any memory apart from incoming
    and outgoing memlets. The DaCe framework cannot analyze these tasklets
    for optimization. 
    """
    def __init__(self,
                 language: Union[str,
                                 dtypes.Language] = dtypes.Language.Python):
        if isinstance(language, str):
            language = dtypes.Language[language]
        self.language = language
        if language != dtypes.Language.Python:
            raise NotImplementedError('Cannot run non-Python tasklet in Python')

    def __enter__(self):
        # TODO: parse and run tasklet
        # TODO: reinstate tasklet simulator
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
