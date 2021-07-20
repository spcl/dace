# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" DaCe Python parsing functionality and entry point to Python frontend. """
from __future__ import print_function
import inspect
import itertools
import copy
import os
import sympy
from typing import Any, Dict, List, Optional, Set, Tuple
import warnings

from dace import symbolic, dtypes
from dace.config import Config
from dace.frontend.python import newast
from dace.sdfg import SDFG
from dace.data import create_datadescriptor, Data

ArgTypes = Dict[str, Data]


def _get_argnames(f) -> List[str]:
    """ Returns a Python function's argument names. """
    try:
        return list(inspect.signature(f).parameters.keys())
    except AttributeError:
        return inspect.getargspec(f).args


def _is_empty(val: Any) -> bool:
    """ Helper function to deal with inspect._empty. """
    return val is inspect._empty


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
            if not _is_empty(pval.default)
        }
        self.symbols = set(k for k, v in self.global_vars.items()
                           if isinstance(v, symbolic.symbol))

        # Add type annotations from decorator arguments (DEPRECATED)
        if self.dec_args:
            warnings.warn(
                'Using decorator arguments for types is deprecated. '
                'Please use type hints on function arguments instead.')
            for arg, pval in zip(self.dec_args,
                                 self.signature.parameters.values()):
                pval._annotation = arg

        # Keep a set of constant arguments to ignore
        self.constant_args = set(
            pname for pname, pval in self.signature.parameters.items()
            if pval.annotation is dtypes.constant)

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
        return self._parse(args, kwargs, strict=strict, save=save)

    def __sdfg__(self, *args, **kwargs) -> SDFG:
        return self._parse(args, kwargs, strict=None, save=False)

    def compile(self, *args, strict=None, save=False, **kwargs):
        """ Convenience function that parses and compiles a DaCe program. """
        sdfg = self._parse(args, kwargs, strict=strict, save=save)

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

    def clear_cache(self):
        """ Force-clear compiled SDFG cache of this program. """
        del self._cache
        self._cache = (None, None, None)

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
                sdfg, {
                    k: create_datadescriptor(v)
                    for k, v in result.items() if k not in self.constant_args
                }))
        return result

    def __call__(self, *args, **kwargs):
        """ Convenience function that parses, compiles, and runs a DaCe 
            program. """
        # Check if SDFG with these argument types and shapes is cached
        if self.methodobj is not None:
            self.global_vars[self.objname] = self.methodobj

        argtypes, arg_mapping, _ = self._get_type_annotations(args, kwargs)

        if self.is_cached(argtypes):
            kwargs.update(arg_mapping)
            self._cache[2].clear_return_values()
            return self._cache[2](
                **self._create_sdfg_args(self._cache[1], args, kwargs))

        # Clear cache to enforce deletion and closure of compiled program
        self.clear_cache()

        # Add classes (that are not defined yet when the method is) to closure
        local_vars = _get_locals_and_globals(self.f)
        self.global_vars.update({
            k: v
            for k, v in local_vars.items()
            if k not in self.global_vars and isinstance(v, type)
        })

        # Parse SDFG
        sdfg = self._parse(args, kwargs)

        # Add named arguments to the call
        kwargs.update(arg_mapping)
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

    def _parse(self, args, kwargs, strict=None, save=False) -> SDFG:
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
        sdfg = self._generate_pdp(args, kwargs, strict=strict)

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

    def _get_type_annotations(
        self, given_args: Tuple[Any], given_kwargs: Dict[str, Any]
    ) -> Tuple[ArgTypes, Dict[str, Any], Dict[str, Any]]:
        """ 
        Obtains types from decorator and/or from type annotations in a function.
        :param given_args: The call-site arguments to the dace.program.
        :param given_kwargs: The call-site keyword arguments to the program.
        :return: A 3-tuple containing (argument type mapping, extra argument 
                 mapping, extra global variable mapping)
        """
        types: ArgTypes = {}
        arg_mapping: Dict[str, Any] = {}
        gvar_mapping: Dict[str, Any] = {}

        # Filter symbols out of given keyword arguments
        given_kwargs = {
            k: v
            for k, v in given_kwargs.items() if k not in self.symbols
        }

        # Make argument mapping to either type annotation, given argument,
        # default argument, or ignore (symbols and constants).
        nargs = len(given_args)
        arg_ind = 0
        for i, (aname, sig_arg) in enumerate(self.signature.parameters.items()):
            if self.objname is not None and aname == self.objname:
                # Skip "self" argument
                continue

            ann = sig_arg.annotation

            # Variable-length arguments: obtain from the remainder of given_*
            if sig_arg.kind is sig_arg.VAR_POSITIONAL:
                vargs = given_args[arg_ind:]

                # If an annotation is given but the argument list is empty, fail
                if not _is_empty(ann) and len(vargs) == 0:
                    raise SyntaxError(
                        'Cannot compile DaCe program with type-annotated '
                        'variable-length (starred) arguments and no given '
                        'parameters. Please compile the program with arguments, '
                        'call it without annotations, or remove the starred '
                        f'arguments (invalid argument name: "{aname}").')

                types.update({
                    f'__arg{j}': create_datadescriptor(varg)
                    for j, varg in enumerate(vargs)
                })
                arg_mapping.update(
                    {f'__arg{j}': varg
                     for j, varg in enumerate(vargs)})
                gvar_mapping[aname] = tuple(f'__arg{j}'
                                            for j in range(len(vargs)))
                # Shift arg_ind to the end
                arg_ind = len(given_args)
            elif sig_arg.kind is sig_arg.VAR_KEYWORD:
                vargs = {
                    k: create_datadescriptor(v)
                    for k, v in given_kwargs.items() if k not in types
                }
                # If an annotation is given but the argument list is empty, fail
                if not _is_empty(ann) and len(vargs) == 0:
                    raise SyntaxError(
                        'Cannot compile DaCe program with type-annotated '
                        'variable-length (starred) keyword arguments and no given '
                        'parameters. Please compile the program with arguments, '
                        'call it without annotations, or remove the starred '
                        f'arguments (invalid argument name: "{aname}").')
                types.update({f'__kwarg_{k}': v for k, v in vargs.items()})
                arg_mapping.update(
                    {f'__kwarg_{k}': given_kwargs[k]
                     for k in vargs.keys()})
                gvar_mapping[aname] = {k: f'__kwarg_{k}' for k in vargs.keys()}
            # END OF VARIABLE-LENGTH ARGUMENTS
            else:
                # Regular arguments (annotations take precedence)
                curarg = None
                is_constant = False
                if not _is_empty(ann):
                    # If constant, use given argument
                    if ann is dtypes.constant:
                        curarg = None
                        is_constant = True
                    else:
                        curarg = ann

                # If no annotation is provided, use given arguments
                if sig_arg.kind is sig_arg.POSITIONAL_ONLY:
                    if arg_ind >= nargs:
                        if curarg is None and not _is_empty(sig_arg.default):
                            curarg = sig_arg.default
                        elif curarg is None:
                            raise SyntaxError(
                                'Not enough arguments given to program (missing '
                                f'argument: "{aname}").')
                    else:
                        if curarg is None:
                            curarg = given_args[arg_ind]
                        arg_ind += 1
                elif sig_arg.kind is sig_arg.POSITIONAL_OR_KEYWORD:
                    if arg_ind >= nargs:
                        if aname not in given_kwargs:
                            if curarg is None and not _is_empty(
                                    sig_arg.default):
                                curarg = sig_arg.default
                            elif curarg is None:
                                raise SyntaxError(
                                    'Not enough arguments given to program (missing '
                                    f'argument: "{aname}").')
                        elif curarg is None:
                            curarg = given_kwargs[aname]
                    else:
                        if curarg is None:
                            curarg = given_args[arg_ind]
                        arg_ind += 1
                elif sig_arg.kind is sig_arg.KEYWORD_ONLY:
                    if aname not in given_kwargs:
                        if curarg is None and not _is_empty(sig_arg.default):
                            curarg = sig_arg.default
                        elif curarg is None:
                            raise SyntaxError(
                                'Not enough arguments given to program (missing '
                                f'argument: "{aname}").')
                    elif curarg is None:
                        curarg = given_kwargs[aname]

                if is_constant:
                    gvar_mapping[aname] = curarg
                    continue  # Skip argument

                # Set type
                types[aname] = create_datadescriptor(curarg)

        # Set __return* arrays from return type annotations
        rettype = self.signature.return_annotation
        if not _is_empty(rettype):
            if isinstance(rettype, tuple):
                for i, subrettype in enumerate(rettype):
                    types[f'__return_{i}'] = create_datadescriptor(subrettype)
            else:
                types['__return'] = create_datadescriptor(rettype)

        return types, arg_mapping, gvar_mapping

    def _generate_pdp(self, args, kwargs, strict=None):
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

        # If exist, obtain type annotations (for compilation)
        argtypes, _, gvars = self._get_type_annotations(args, kwargs)

        # Parse argument types from call
        if len(self.argnames) > 0:
            if not argtypes:
                if not args and not kwargs:
                    raise SyntaxError(
                        'Compiling DaCe programs requires static types. '
                        'Please provide type annotations on the function, '
                        'or add sample arguments to the compilation call.')

                # Parse compilation arguments
                argtypes = {
                    k: create_datadescriptor(v)
                    for k, v in itertools.chain(self.default_args.items(
                    ), zip(self.argnames, args), kwargs.items())
                }
                if len(argtypes) != len(self.argnames):
                    raise SyntaxError(
                        'Number of arguments must match parameters '
                        f'(expecting {self.argnames}, got {list(argtypes.keys())})'
                    )

        for k, v in argtypes.items():
            if v.transient:  # Arguments to (nested) SDFGs cannot be transient
                v_cpy = copy.deepcopy(v)
                v_cpy.transient = False
                argtypes[k] = v_cpy

        #############################################

        # Parse allowed global variables
        # (for inferring types and values in the DaCe program)
        global_vars = copy.copy(self.global_vars)

        # Remove None arguments and make into globals that can be folded
        for k, v in argtypes.items():
            if v.dtype.type is None:
                global_vars[k] = None
        argtypes = {
            k: v
            for k, v in argtypes.items() if v.dtype.type is not None
        }

        # Set module aliases to point to their actual names
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

        # Add constant arguments to global_vars
        global_vars.update(gvars)

        # Allow SDFGs and DaceProgram objects
        # NOTE: These are the globals AT THE TIME OF INVOCATION, NOT DEFINITION
        other_sdfgs = {
            k: v
            for k, v in _get_locals_and_globals(dace_func).items()
            if isinstance(v, (SDFG, DaceProgram))
        }

        # Parse AST to create the SDFG
        sdfg = newast.parse_dace_program(dace_func,
                                         self.name,
                                         argtypes,
                                         global_vars,
                                         modules,
                                         other_sdfgs,
                                         self.dec_kwargs,
                                         strict=strict)

        # Set SDFG argument names, filtering out constants
        sdfg.arg_names = [a for a in self.argnames if a in argtypes]

        return sdfg
