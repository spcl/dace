# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Python decorators for DaCe functions. """

from __future__ import print_function

from dace import dtypes
from dace.dtypes import paramdec
from dace.frontend.python import parser
from typing import Callable

#############################################

# Type hint specifically for the @dace.program decorator
paramdec_program: Callable[..., Callable[..., parser.DaceProgram]] = paramdec


@paramdec_program
def program(f, *args, **kwargs) -> parser.DaceProgram:
    """ DaCe program, entry point to a data-centric program. """

    # Parses a python @dace.program function and returns an object that can
    # be translated
    return parser.DaceProgram(f, args, kwargs)


function = program

# Internal DaCe decorators, these are not actually run, but rewritten


# Dataflow constructs
@paramdec
def map(f, rng):
    """ A Map is representation of parallel execution, containing
        an integer set (Python range) for which its contents are run 
        concurrently.
        :param rng: The map's range.
    """
    pass


@paramdec
def consume(f, stream, pes):
    """ Consume is a scope, like `Map`, that creates parallel execution.
        Unlike `Map`, it creates a producer-consumer relationship between an
        input stream and the contents. The contents are run by the given number
        of processing elements, who will try to pop elements from the input
        stream until a given quiescence condition is reached. 
        :param stream: The stream to pop from.
        :param pes: The number of processing elements to use.
    """
    pass


def tasklet(f):
    """ A general procedure that cannot access any memory apart from incoming
        and outgoing memlets. The DaCe framework cannot analyze these tasklets
        for optimization. """
    pass


# Control-flow constructs
@paramdec
def iterate(f, rng):
    """ A decorator version of a for loop, with a range of `rng`.
        :param rng: The range of the for loop.
    """
    pass


@paramdec
def loop(f, cond):
    """ A decorator version of a while loop, with a looping condition `cond`.
        :param cond: The condition of the while loop.
    """
    pass


@paramdec
def conditional(f, cond):
    """ A decorator version of conditional execution, with an if-condition 
        `cond`.
        :param cond: The condition of the branch.
    """
    pass
