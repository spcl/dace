# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" DaCe Python parsing functionality and entry point to Python frontend. """
from __future__ import print_function
import collections
import inspect
import copy
import os
import sympy
from typing import Any, Dict, Optional, Set, Tuple

from dace import symbolic, dtypes
from dace.config import Config
from dace.frontend.python import newast
from dace.sdfg import SDFG
from dace.data import create_datadescriptor, Data

ArgTypes = Dict[str, Data]


def get_type_annotations(f,
                         f_argnames,
                         decorator_args,
                         method: bool = False) -> ArgTypes:
    """ Obtains types from decorator or from type annotations in a function. 
    """
    type_annotations = {}
    if hasattr(f, '__annotations__'):
        type_annotations.update(f.__annotations__)

    # Type annotation conditions
    has_args = len(decorator_args) > 0
    has_annotations = len(type_annotations) > 0

    # Set __return* arrays from return type annotations
    if 'return' in type_annotations:
        rettype = type_annotations['return']
        if isinstance(rettype, tuple):
            for i, subrettype in enumerate(rettype):
                type_annotations[f'__return_{i}'] = subrettype
        else:
            type_annotations['__return'] = rettype
        del type_annotations['return']

    # If both arguments and annotations are given, annotations take precedence
    if has_args and has_annotations:
        has_args = False

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
        filtered = {
            a
            for a in type_annotations.keys() if not a.startswith('__return')
        }
        if len(filtered) != len(f_argnames):
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


def infer_symbols_from_datadescriptor(sdfg: SDFG, args: Dict[str, Any],
                                      exclude: Optional[Set[str]] = None) -> \
        Dict[str, Any]:
    """
    Infers the values of SDFG symbols (not given as arguments) from the shapes
    and strides of input arguments (e.g., arrays).
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
            symbolic_values = list(desc.shape) + list(
                getattr(desc, 'strides', []))
            given_values = list(arg_val.shape)
            given_strides = []
            if hasattr(arg_val, 'strides'):
                # NumPy arrays use bytes in strides
                factor = getattr(arg_val, 'itemsize', 1)
                given_strides = [s // factor for s in arg_val.strides]
            given_values += given_strides

            for sym_dim, real_dim in zip(symbolic_values, given_values):
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

    # Remove __SOLVE_ prefix
    return {str(k)[8:]: v for k, v in result.items()}


class DaceProgram:
    """ A data-centric program object, obtained by decorating a function with
        ``@dace.program``. """
    def __init__(self, f, args, kwargs, auto_optimize, device, method=False):
        from dace.codegen import compiled_sdfg  # Avoid import loops

        self.f = f
        self.dec_args = args
        self.dec_kwargs = kwargs
        self.name = f.__name__
        self.argnames = _get_argnames(f)
        if method:
            self.objname = self.argnames[0]
            self.argnames = self.argnames[1:]
        else:
            self.objname = None
        self.auto_optimize = auto_optimize
        self.device = device
        self._methodobj: Any = None  #: Object whose method this program is

        self.global_vars = _get_locals_and_globals(f)
        self.signature = inspect.signature(f)
        self.default_args = {
            pname: pval.default
            for pname, pval in self.signature.parameters.items()
            if pval.default is not inspect._empty
        }

        if self.argnames is None:
            self.argnames = []

        # Cache SDFG with last used arguments
        self._cache: Tuple[ArgTypes, SDFG,
                           compiled_sdfg.CompiledSDFG] = (None, None, None)

    def _auto_optimize(self,
                       sdfg: SDFG,
                       symbols: Dict[str, int] = None) -> SDFG:
        """ Invoke automatic optimization heuristics on internal program. """
        # Avoid import loop
        from dace.transformation.auto import auto_optimize as autoopt
        return autoopt.auto_optimize(sdfg, self.device, symbols=symbols)

    def to_sdfg(self, *args, strict=None, save=False, **kwargs) -> SDFG:
        """ Parses the DaCe function into an SDFG. """
        return self.parse(args, kwargs, strict=strict, save=save)

    def __sdfg__(self, *args, **kwargs) -> SDFG:
        return self.parse(args, kwargs, strict=None, save=False)

    def compile(self, *args, strict=None, save=False, **kwargs):
        """ Convenience function that parses and compiles a DaCe program. """
        sdfg = self.parse(args, kwargs, strict=strict, save=save)

        # Invoke auto-optimization as necessary
        if Config.get_bool('optimizer', 'autooptimize') or self.auto_optimize:
            sdfg = self._auto_optimize(sdfg)

        return sdfg.compile()

    def is_cached(self, argtypes: ArgTypes) -> bool:
        """
        Returns True if the given arguments exist in the compiled SDFG cache.
        """
        if self._cache[0] is None or self._cache[0].keys() != argtypes.keys():
            return False
        for k, v in self._cache[0].items():
            if not v.is_equivalent(argtypes[k]):
                return False
        return True

    @property
    def methodobj(self) -> Any:
        return self._methodobj

    @methodobj.setter
    def methodobj(self, new_obj: Any):
        self._methodobj = new_obj
        # Clear cache upon changing parent object
        del self._cache
        self._cache = (None, None, None)

    def _create_sdfg_args(self, sdfg: SDFG, args: Tuple[Any],
                          kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # Start with default arguments, then add other arguments
        result = {**self.default_args}
        # Reconstruct keyword arguments
        result.update({aname: arg for aname, arg in zip(self.argnames, args)})
        result.update(kwargs)

        # Update arguments with symbols in data shapes
        result.update(
            infer_symbols_from_datadescriptor(
                sdfg, {k: create_datadescriptor(v)
                       for k, v in kwargs.items()}))
        return result

    def __call__(self, *args, **kwargs):
        """ Convenience function that parses, compiles, and runs a DaCe 
            program. """
        # Check if SDFG with these argument types and shapes is cached
        if self.methodobj is not None:
            self.global_vars[self.objname] = self.methodobj

        argtypes = get_type_annotations(self.f, self.argnames, args,
                                        self.methodobj is not None)
        if self.is_cached(argtypes):
            self._cache[2].clear_return_values()
            return self._cache[2](
                **self._create_sdfg_args(self._cache[1], args, kwargs))

        # Clear cache to enforce deletion and closure of compiled program
        del self._cache

        # Add classes (that are not defined yet when the method is) to closure
        local_vars = _get_locals_and_globals(self.f)
        self.global_vars.update({
            k: v
            for k, v in local_vars.items()
            if k not in self.global_vars and isinstance(v, type)
        })

        # Parse SDFG
        sdfg = self.parse(args, kwargs)

        # Add named arguments to the call
        sdfg_args = self._create_sdfg_args(sdfg, args, kwargs)

        # Allow CLI to prompt for optimizations
        if Config.get_bool('optimizer', 'transform_on_call'):
            sdfg = sdfg.optimize()

        # Invoke auto-optimization as necessary
        if Config.get_bool('optimizer', 'autooptimize') or self.auto_optimize:
            sdfg = self._auto_optimize(sdfg, symbols=sdfg_args)

        # Compile SDFG (note: this is done after symbol inference due to shape
        # altering transformations such as Vectorization)
        binaryobj = sdfg.compile()

        self._cache = (argtypes, sdfg, binaryobj)

        # Call SDFG
        result = binaryobj(**sdfg_args)

        return result

    def parse(self, args, kwargs, strict=None, save=False) -> SDFG:
        """ 
        Try to parse a DaceProgram object and return the `dace.SDFG` object
        that corresponds to it.
        :param function: DaceProgram object (obtained from the ``@dace.program``
                        decorator).
        :param args: The given arguments to the function.
        :param kwargs: The given keyword arguments to the function.
        :param strict: Whether to apply strict transformations or not (None
                       uses configuration-defined value). 
        :param save: If True, saves the generated SDFG to 
                    ``_dacegraphs/program.sdfg`` after parsing.
        :return: The generated SDFG object.
        """
        # Avoid import loop
        from dace.sdfg.analysis import scalar_to_symbol as scal2sym
        from dace.transformation import helpers as xfh

        # Obtain DaCe program as SDFG
        sdfg = self.generate_pdp(args, kwargs, strict=strict)

        # Set argument names
        sdfg.arg_names = self.argnames

        # Apply strict transformations automatically
        if (strict == True or (strict is None and Config.get_bool(
                'optimizer', 'automatic_strict_transformations'))):

            # Promote scalars to symbols as necessary
            promoted = scal2sym.promote_scalars_to_symbols(sdfg)
            if Config.get_bool('debugprint') and len(promoted) > 0:
                print('Promoted scalars {%s} to symbols.' %
                      ', '.join(p for p in sorted(promoted)))

            sdfg.apply_strict_transformations()

            # Split back edges with assignments and conditions to allow richer
            # control flow detection in code generation
            xfh.split_interstate_edges(sdfg)

        # Save the SDFG. Skip this step if running from a cached SDFG, as
        # it might overwrite the cached SDFG.
        if not Config.get_bool('compiler', 'use_cache') and save:
            sdfg.save(os.path.join('_dacegraphs', 'program.sdfg'))

        # Validate SDFG
        sdfg.validate()

        return sdfg

    def generate_pdp(self, args, kwargs, strict=None):
        """ Generates the parsed AST representation of a DaCe program.
            :param args: The given arguments to the program.
            :param kwargs: The given keyword arguments to the program.
            :param strict: Whether to apply strict transforms when parsing 
                           nested dace programs.
            :return: A 2-tuple of (program, modules), where `program` is a
                     `dace.astnodes._ProgramNode` representing the parsed DaCe 
                     program, and `modules` is a dictionary mapping imported 
                     module names to their actual module names (for maintaining
                     import aliases).
        """
        dace_func = self.f
        dargs = self.dec_args

        # If exist, obtain type annotations (for compilation)
        argtypes = get_type_annotations(dace_func, self.argnames, dargs,
                                        self.methodobj is not None)

        # Parse argument types from call
        if len(inspect.getfullargspec(dace_func).args) > 0:
            if not argtypes:
                if not args:
                    raise SyntaxError(
                        'Compiling DaCe programs requires static types. '
                        'Please provide type annotations on the function, '
                        'or add sample arguments to the compilation call.')

                # Parse compilation arguments
                if len(args) != len(self.argnames):
                    raise SyntaxError(
                        'Number of keyword arguments must match parameters '
                        '(expecting %d)' % len(self.argnames))
                argtypes = {
                    k: create_datadescriptor(v)
                    for k, v in zip(self.argnames, args)
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
            for _, v in global_vars.items() if isinstance(v, symbolic.symbol)
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
                                         self.name,
                                         argtypes,
                                         global_vars,
                                         modules,
                                         other_sdfgs,
                                         self.dec_kwargs,
                                         strict=strict)
