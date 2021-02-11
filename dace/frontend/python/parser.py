# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" DaCe Python parsing functionality and entry point to Python frontend. """
from __future__ import print_function
import inspect
import copy
import os
import sympy
from typing import Any, Dict, Optional, Set

from dace import symbolic, dtypes
from dace.config import Config
from dace.frontend.python import newast
from dace.sdfg import SDFG
from dace.data import create_datadescriptor


def _get_type_annotations(f, f_argnames, decorator_args):
    """ Obtains types from decorator or from type annotations in a function. 
    """
    type_annotations = {}
    if hasattr(f, '__annotations__'):
        type_annotations.update(f.__annotations__)

    # Type annotation conditions
    has_args = len(decorator_args) > 0
    has_annotations = len(type_annotations) > 0
    if 'return' in type_annotations:
        raise TypeError('DaCe programs do not have a return type')
    if has_args and has_annotations:
        raise SyntaxError('DaCe programs can only have decorator arguments ' +
                          '(\'@dace.program(...)\') or type annotations ' +
                          '(\'def program(arr: type, ...)\'), but not both')

    # Alert if there are any discrepancies between annotations and arguments
    if has_args:
        # Make sure all arguments are annotated
        if len(decorator_args) != len(f_argnames):
            raise SyntaxError('Decorator arguments must match number of DaCe ' +
                              'program parameters (expecting ' +
                              str(len(f_argnames)) + ')')
        # Return arguments and their matched decorator annotation
        return {
            k: create_datadescriptor(v)
            for k, v in zip(f_argnames, decorator_args)
        }
    elif has_annotations:
        # Make sure all arguments are annotated
        if len(type_annotations) != len(f_argnames):
            raise SyntaxError(
                'Either none or all DaCe program parameters must ' +
                'have type annotations')
    return {k: create_datadescriptor(v) for k, v in type_annotations.items()}


def _get_argnames(f):
    """ Returns a Python function's argument names. """
    try:
        return inspect.getfullargspec(f).args
    except AttributeError:
        return inspect.getargspec(f).args


def _compile_module(s, name='<string>'):
    """ Compiles a string representing a python module (file or code) and
        returns the resulting global objects as a dictionary mapping name->val.
        :param name: Optional name for better error message handling.
    """

    gen_module = {}
    code = compile(s, name, 'exec')
    exec(code, gen_module)
    return gen_module


def parse_from_file(filename, *compilation_args):
    """ Try to parse all DaCe programs in `filename` and return a list of
        obtained SDFGs. Raises exceptions in case of compilation errors.
        Also accepts optional compilation arguments containing types and symbol
        values.
    """

    with open(filename, 'r') as f:
        code = f.read()

    mod = _compile_module(code, filename)

    programs = [
        program for program in mod.values() if isinstance(program, DaceProgram)
    ]

    return [parse_function(p, *compilation_args) for p in programs]


def parse_from_function(function, *compilation_args, strict=None):
    """ Try to parse a DaceProgram object and return the `dace.SDFG` object
        that corresponds to it.
        :param function: DaceProgram object (obtained from the `@dace.program`
                         decorator).
        :param compilation_args: Various compilation arguments e.g. dtypes.
        :param strict: Whether to apply strict transformations or not (None
                       uses configuration-defined value). 
        :return: The generated SDFG object.
    """
    # Avoid import loop
    from dace.sdfg.analysis import scalar_to_symbol as scal2sym
    from dace.transformation import helpers as xfh

    if not isinstance(function, DaceProgram):
        raise TypeError(
            'Function must be of type dace.frontend.python.DaceProgram')

    # Obtain DaCe program as SDFG
    sdfg = function.generate_pdp(*compilation_args, strict=strict)

    # Apply strict transformations automatically
    if (strict == True or
        (strict is None
         and Config.get_bool('optimizer', 'automatic_strict_transformations'))):

        # Promote scalars to symbols as necessary
        promoted = scal2sym.promote_scalars_to_symbols(sdfg)
        if Config.get_bool('debugprint') and len(promoted) > 0:
            print('Promoted scalars {%s} to symbols.' %
                  ', '.join(p for p in sorted(promoted)))

        sdfg.apply_strict_transformations()

        # Split back edges with assignments and conditions to allow richer
        # control flow detection in code generation
        xfh.split_interstate_edges(sdfg)

    # Save the SDFG (again). Skip this step if running from a cached SDFG, as
    # it might overwrite the cached SDFG.
    if not Config.get_bool('compiler', 'use_cache'):
        sdfg.save(os.path.join('_dacegraphs', 'program.sdfg'))

    # Validate SDFG
    sdfg.validate()

    return sdfg


def _get_locals_and_globals(f):
    """ Retrieves a list of local and global variables for the function ``f``.
        This is used to retrieve variables around and defined before  @dace.programs for adding symbols and constants.
    """
    result = {}
    # Update globals, then locals
    result.update(f.__globals__)
    # grab the free variables (i.e. locals)
    if f.__closure__ is not None:
        result.update({
            k: v
            for k, v in zip(f.__code__.co_freevars,
                            [x.cell_contents for x in f.__closure__])
        })

    return result


def infer_symbols_from_shapes(sdfg: SDFG, args: Dict[str, Any],
                              exclude: Optional[Set[str]] = None) -> \
        Dict[str, Any]:
    """
    Infers the values of SDFG symbols (not given as arguments) from the shapes
    of input arguments (e.g., arrays).
    :param sdfg: The SDFG that is being called.
    :param args: A dictionary mapping from current argument names to their
                 values. This may also include symbols.
    :param exclude: An optional set of symbols to ignore on inference.
    :return: A dictionary mapping from symbol names that are not in ``args``
             to their inferred values.
    :raise ValueError: If symbol values are ambiguous.
    """
    exclude = exclude or set()
    exclude = set(symbolic.symbol(s) for s in exclude)
    equations = []
    symbols = set()
    # Collect equations and symbols from arguments and shapes
    for arg_name, arg_val in args.items():
        if arg_name in sdfg.arrays:
            desc = sdfg.arrays[arg_name]
            if not hasattr(desc, 'shape') or not hasattr(arg_val, 'shape'):
                continue
            symbolic_shape = desc.shape
            given_shape = arg_val.shape

            for sym_dim, real_dim in zip(symbolic_shape, given_shape):
                repldict = {}
                for sym in symbolic.symlist(sym_dim).values():
                    newsym = symbolic.symbol('__SOLVE_' + str(sym))
                    if str(sym) in args:
                        exclude.add(newsym)
                    else:
                        symbols.add(newsym)
                        exclude.add(sym)
                    repldict[sym] = newsym

                # Replace symbols with __SOLVE_ symbols so as to allow
                # the same symbol in the called SDFG
                if repldict:
                    sym_dim = sym_dim.subs(repldict)

                equations.append(sym_dim - real_dim)

    if len(symbols) == 0:
        return {}

    # Solve for all at once
    results = sympy.solve(equations, *symbols, dict=True, exclude=exclude)
    if len(results) > 1:
        raise ValueError('Ambiguous values for symbols in inference. '
                         'Options: %s' % str(results))
    if len(results) == 0:
        raise ValueError('Cannot infer values for symbols in inference.')

    result = results[0]
    if not result:
        raise ValueError('Cannot infer values for symbols in inference.')

    # Fast path (unnecessary)
    # # For each symbol in each dimension, try to solve an equation
    # for sym_dim, real_dim in zip(symbolic_shape, given_shape):
    #     for sym in symbolic.symlist(sym_dim):
    #         if sym in inferred_syms and symval != inferred_syms[sym]:
    #             raise ValueError('Ambiguous value for symbol %s in argument '
    #                              '%s: can be either %d or %d' % (
    #                 sym, arg_name, inferred_syms[sym], symval))

    # Remove __SOLVE_ prefix
    return {str(k)[8:]: v for k, v in result.items()}


class DaceProgram:
    """ A data-centric program object, obtained by decorating a function with
        `@dace.program`. """
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs
        self._name = f.__name__
        self.argnames = _get_argnames(f)

        global_vars = _get_locals_and_globals(f)

        self.global_vars = {
            k: v
            for k, v in global_vars.items() if dtypes.isallowed(v, allow_recursive=True)
        }
        if self.argnames is None:
            self.argnames = []

    @property
    def name(self):
        return self._name

    def to_sdfg(self, *args, strict=None) -> SDFG:
        """ Parses the DaCe function into an SDFG. """
        return parse_from_function(self, *args, strict=strict)

    def compile(self, *args, strict=None):
        """ Convenience function that parses and compiles a DaCe program. """
        sdfg = parse_from_function(self, *args, strict=strict)
        return sdfg.compile()

    def __call__(self, *args, **kwargs):
        """ Convenience function that parses, compiles, and runs a DaCe 
            program. """
        # Parse SDFG
        sdfg = parse_from_function(self, *args)

        # Add named arguments to the call
        kwargs.update({aname: arg for aname, arg in zip(self.argnames, args)})

        # Update arguments with symbols in data shapes
        kwargs.update(infer_symbols_from_shapes(sdfg, kwargs))

        # Allow CLI to prompt for optimizations
        if Config.get_bool('optimizer', 'transform_on_call'):
            sdfg = sdfg.optimize()

        # Compile SDFG (note: this is done after symbol inference due to shape
        # altering transformations such as Vectorization)
        binaryobj = sdfg.compile()

        return binaryobj(**kwargs)

    def generate_pdp(self, *compilation_args, strict=None):
        """ Generates the parsed AST representation of a DaCe program.
            :param compilation_args: Various compilation arguments e.g., dtypes.
            :param strict: Whether to apply strict transforms when parsing nested dace programs.
            :return: A 2-tuple of (program, modules), where `program` is a
                     `dace.astnodes._ProgramNode` representing the parsed DaCe 
                     program, and `modules` is a dictionary mapping imported 
                     module names to their actual module names (for maintaining
                     import aliases).
        """
        dace_func = self.f
        args = self.args

        # If exist, obtain type annotations (for compilation)
        argtypes = _get_type_annotations(dace_func, self.argnames, args)

        # Parse argument types from call
        if len(inspect.getfullargspec(dace_func).args) > 0:
            if not argtypes:
                if not compilation_args:
                    raise SyntaxError(
                        'DaCe program compilation requires either type annotations '
                        'or arrays')

                # Parse compilation arguments
                if len(compilation_args) != len(self.argnames):
                    raise SyntaxError(
                        'Arguments must match DaCe program parameters (expecting '
                        '%d)' % len(self.argnames))
                argtypes = {
                    k: create_datadescriptor(v)
                    for k, v in zip(self.argnames, compilation_args)
                }
        for k, v in argtypes.items():
            if v.transient:  # Arguments to (nested) SDFGs cannot be transient
                v_cpy = copy.deepcopy(v)
                v_cpy.transient = False
                argtypes[k] = v_cpy
        #############################################

        # Parse allowed global variables
        # (for inferring types and values in the DaCe program)
        global_vars = copy.copy(self.global_vars)

        modules = {
            k: v.__name__
            for k, v in global_vars.items() if dtypes.ismodule(v)
        }
        modules['builtins'] = ''

        # Add symbols as globals with their actual names (sym_0 etc.)
        global_vars.update({
            v.name: v
            for k, v in global_vars.items() if isinstance(v, symbolic.symbol)
        })
        for argtype in argtypes.values():
            global_vars.update({v.name: v for v in argtype.free_symbols})

        # Allow SDFGs and DaceProgram objects
        # NOTE: These are the globals AT THE TIME OF INVOCATION, NOT DEFINITION
        other_sdfgs = {
            k: v
            for k, v in _get_locals_and_globals(dace_func).items()
            if isinstance(v, (SDFG, DaceProgram))
        }

        # Parse AST to create the SDFG
        return newast.parse_dace_program(dace_func,
                                         argtypes,
                                         global_vars,
                                         modules,
                                         other_sdfgs,
                                         self.kwargs,
                                         strict=strict)
