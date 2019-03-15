""" Python decorators for DaCe functions. """

from __future__ import print_function
from functools import wraps

from dace import types
from dace.frontend.python import parser


def paramdec(dec):
    """ Parameterized decorator meta-decorator. Enables using `@decorator`,
        `@decorator()`, and `@decorator(...)` with the same function. """

    @wraps(dec)
    def layer(*args, **kwargs):

        # Allows the use of @decorator, @decorator(), and @decorator(...)
        if len(kwargs) == 0 and len(args) == 1 and callable(
                args[0]) and not isinstance(args[0], types.typeclass):
            return dec(*args, **kwargs)

        @wraps(dec)
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


#############################################


@paramdec
def program(f, *args, **kwargs):
    """ DaCe program, entry point to a data-centric program. """

    # Parses a python @dace.program function and returns an object that can
    # be translated
    return parser.DaceProgram(f, args, kwargs)


#############################################


@paramdec
def external_function(f, **alternative_implementations):
    """ External functions that may be called within a DaCe program. """
    return types._external_function(f, alternative_implementations)


# Internal DaCe decorators, these are not actually run, but rewritten


# Dataflow constructs
@paramdec
def map(f, rng):
    """ A Map is representation of parallel execution, containing
        an integer set (Python range) for which its contents are run 
        concurrently.
        @param rng: The map's range.
    """
    return None


@paramdec
def consume(f, stream, pes):
    """ Consume is a scope, like `Map`, that creates parallel execution.
        Unlike `Map`, it creates a producer-consumer relationship between an
        input stream and the contents. The contents are run by the given number
        of processing elements, who will try to pop elements from the input
        stream until a given quiescence condition is reached. 
        @param stream: The stream to pop from.
        @param pes: The number of processing elements to use.
    """
    return None


def tasklet(f):
    """ A general procedure that cannot access any memory apart from incoming
        and outgoing memlets. The DaCe framework cannot analyze these tasklets
        for optimization. """
    return None


# Control-flow constructs
@paramdec
def iterate(f, rng):
    """ A decorator version of a for loop, with a range of `rng`.
        @param rng: The range of the for loop.
    """
    return None


@paramdec
def loop(f, cond):
    """ A decorator version of a while loop, with a looping condition `cond`.
        @param cond: The condition of the while loop.
    """
    return None


@paramdec
def conditional(f, cond):
    """ A decorator version of conditional execution, with an if-condition 
        `cond`.
        @param cond: The condition of the branch.
    """
    return None
