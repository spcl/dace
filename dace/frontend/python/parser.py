# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" DaCe Python parsing functionality and entry point to Python frontend. """
from dataclasses import dataclass
import inspect
import itertools
import copy
import os
import sympy
from typing import Any, Callable, Dict, List, Optional, Set, Sequence, Tuple
import warnings

from dace import symbolic, dtypes
from dace.config import Config
from dace.frontend.python import (newast, common as pycommon, cached_program, preprocessing)
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


def _get_cell_contents_or_none(cell):
    try:
        return cell.cell_contents
    except ValueError:  # Empty cell
        return None


def _get_locals_and_globals(f):
    """ Retrieves a list of local and global variables for the function ``f``.
        This is used to retrieve variables around and defined before  @dace.programs for adding symbols and constants.
    """
    result = {}
    # Update globals, then locals
    result.update(f.__globals__)
    # grab the free variables (i.e. locals)
    if f.__closure__ is not None:
        result.update(
            {k: v
             for k, v in zip(f.__code__.co_freevars, [_get_cell_contents_or_none(x) for x in f.__closure__])})

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
            symbolic_values = list(desc.shape) + list(getattr(desc, 'strides', []))
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
        raise ValueError('Ambiguous values for symbols in inference. ' 'Options: %s' % str(results))
    if len(results) == 0:
        raise ValueError('Cannot infer values for symbols in inference.')

    result = results[0]
    if not result:
        raise ValueError('Cannot infer values for symbols in inference.')

    # Remove __SOLVE_ prefix
    return {str(k)[8:]: v for k, v in result.items()}


class DaceProgram(pycommon.SDFGConvertible):
    """ A data-centric program object, obtained by decorating a function with
        ``@dace.program``. """
    def __init__(self, f, args, kwargs, auto_optimize, device, constant_functions=False, method=False):
        from dace.codegen import compiled_sdfg  # Avoid import loops

        self.f = f
        self.dec_args = args
        self.dec_kwargs = kwargs
        self.resolve_functions = constant_functions
        self.argnames = _get_argnames(f)
        if method:
            self.objname = self.argnames[0]
            self.argnames = self.argnames[1:]
        else:
            self.objname = None
        self.auto_optimize = auto_optimize
        self.device = device
        self._methodobj: Any = None  #: Object whose method this program is
        self.validate: bool = True  #: Whether to validate on code generation

        self.global_vars = _get_locals_and_globals(f)
        self.signature = inspect.signature(f)
        self.default_args = {
            pname: pval.default
            for pname, pval in self.signature.parameters.items() if not _is_empty(pval.default)
        }
        self.symbols = set(k for k, v in self.global_vars.items() if isinstance(v, symbolic.symbol))
        self.closure_arg_mapping: Dict[str, Callable[[], Any]] = {}
        self.resolver: pycommon.SDFGClosure = None

        # Add type annotations from decorator arguments (DEPRECATED)
        if self.dec_args:
            warnings.warn('Using decorator arguments for types is deprecated. '
                          'Please use type hints on function arguments instead.')
            for arg, pval in zip(self.dec_args, self.signature.parameters.values()):
                pval._annotation = arg

        # Keep a set of constant arguments to ignore
        self.constant_args = set(pname for pname, pval in self.signature.parameters.items()
                                 if pval.annotation is dtypes.constant)

        if self.argnames is None:
            self.argnames = []

        # Cache SDFGs with last used arguments
        self._cache = cached_program.DaceProgramCache(self._eval_closure)
        # These sets fill up after the first parsing of the program and stay
        # the same unless the argument types change
        self.closure_array_keys: Set[str] = set()
        self.closure_constant_keys: Set[str] = set()

    def _auto_optimize(self, sdfg: SDFG, symbols: Dict[str, int] = None) -> SDFG:
        """ Invoke automatic optimization heuristics on internal program. """
        # Avoid import loop
        from dace.transformation.auto import auto_optimize as autoopt
        return autoopt.auto_optimize(sdfg, self.device, symbols=symbols)

    def to_sdfg(self, *args, coarsen=None, save=False, validate=False, use_cache=False, **kwargs) -> SDFG:
        """ Parses the DaCe function into an SDFG. """
        if use_cache:
            # Update global variables with current closure
            self.global_vars = _get_locals_and_globals(self.f)

            # Move "self" from an argument into the closure
            if self.methodobj is not None:
                self.global_vars[self.objname] = self.methodobj

            argtypes, arg_mapping, constant_args = self._get_type_annotations(args, kwargs)

            # Add constant arguments to globals for caching
            self.global_vars.update(constant_args)

            # Check cache for already-parsed SDFG
            cachekey = self._cache.make_key(argtypes, self.closure_array_keys, self.closure_constant_keys,
                                            constant_args)

            if self._cache.has(cachekey):
                entry = self._cache.get(cachekey)
                return entry.sdfg

        sdfg = self._parse(args, kwargs, coarsen=coarsen, save=save, validate=validate)

        if use_cache:
            # Add to cache
            self._cache.add(cachekey, sdfg, None)

        return sdfg

    def __sdfg__(self, *args, **kwargs) -> SDFG:
        return self._parse(args, kwargs, coarsen=None, save=False, validate=False)

    def compile(self, *args, coarsen=None, save=False, **kwargs):
        """ Convenience function that parses and compiles a DaCe program. """
        sdfg = self._parse(args, kwargs, coarsen=coarsen, save=save)

        # Invoke auto-optimization as necessary
        if Config.get_bool('optimizer', 'autooptimize') or self.auto_optimize:
            sdfg = self._auto_optimize(sdfg)

        return sdfg.compile(validate=self.validate)

    @property
    def methodobj(self) -> Any:
        return self._methodobj

    @methodobj.setter
    def methodobj(self, new_obj: Any):
        self._methodobj = new_obj

    @property
    def name(self) -> str:
        """ Returns a unique name for this program. """
        result = ''
        if self.f.__module__ is not None and self.f.__module__ != '__main__':
            result += self.f.__module__.replace('.', '_') + '_'
        if self._methodobj is not None:
            result += type(self._methodobj).__name__ + '_'
        return result + self.f.__name__

    def __sdfg_signature__(self) -> Tuple[Sequence[str], Sequence[str]]:
        return self.argnames, self.constant_args

    def __sdfg_closure__(self, reevaluate: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """ 
        Returns the closure arrays of the SDFG represented by the dace 
        program as a mapping between array name and the corresponding value.
        :param reevaluate: If given, re-evaluates closure elements based on the
                           input mapping (keys: array names, values: expressions
                           to evaluate). Otherwise, re-evaluates 
                           ``self.closure_arg_mapping``.
        :return: A dictionary mapping between a name in the closure and the 
                 currently evaluated value.
        """
        # Move "self" from an argument into the closure
        if self.methodobj is not None:
            self.global_vars[self.objname] = self.methodobj

        if reevaluate is None:
            result = {k: v() for k, v in self.closure_arg_mapping.items()}
            if self.resolver is not None:
                result.update({k: v[1] for k, v in self.resolver.callbacks.items()})
            return result
        else:
            return {k: eval(v, self.global_vars) if isinstance(v, str) else v for k, v in reevaluate.items()}

    def closure_resolver(self, constant_args, parent_closure=None):
        # Parse allowed global variables
        # (for inferring types and values in the DaCe program)
        global_vars = copy.copy(self.global_vars)

        # If exist, obtain compile-time constants
        gvars = {}
        if constant_args is not None:
            gvars = {self.argnames[i]: v for i, v in constant_args.items() if isinstance(i, int)}
            gvars.update({k: v for k, v in constant_args.items() if not isinstance(k, int)})
        global_vars = {k: v for k, v in global_vars.items() if k not in self.argnames}

        # Move "self" from an argument into the closure
        if self.methodobj is not None:
            global_vars[self.objname] = self.methodobj

        # Set module aliases to point to their actual names
        modules = {k: v.__name__ for k, v in global_vars.items() if dtypes.ismodule(v)}
        modules['builtins'] = ''

        # Add symbols as globals with their actual names (sym_0 etc.)
        global_vars.update({v.name: v for _, v in global_vars.items() if isinstance(v, symbolic.symbol)})

        # Add constant arguments to global_vars
        global_vars.update(gvars)

        # Parse AST to create the SDFG
        _, closure = preprocessing.preprocess_dace_program(self.f, {},
                                                           global_vars,
                                                           modules,
                                                           resolve_functions=self.resolve_functions,
                                                           parent_closure=parent_closure)
        return closure

    def _eval_closure(self, arg: str, extra_constants: Optional[Dict[str, Any]] = None) -> Any:
        extra_constants = extra_constants or {}
        if arg in self.closure_arg_mapping:
            return self.closure_arg_mapping[arg]()
        return eval(arg, self.global_vars, extra_constants)

    def _create_sdfg_args(self, sdfg: SDFG, args: Tuple[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # Start with default arguments, then add other arguments
        result = {**self.default_args}
        # Reconstruct keyword arguments
        result.update({aname: arg for aname, arg in zip(self.argnames, args)})
        result.update(kwargs)

        # Add closure arguments to the call
        result.update(self.__sdfg_closure__())

        # Update arguments with symbols in data shapes
        result.update(
            infer_symbols_from_datadescriptor(
                sdfg, {k: create_datadescriptor(v)
                       for k, v in result.items() if k not in self.constant_args}))
        return result

    def __call__(self, *args, **kwargs):
        """ Convenience function that parses, compiles, and runs a DaCe 
            program. """
        # Update global variables with current closure
        self.global_vars = _get_locals_and_globals(self.f)

        # Move "self" from an argument into the closure
        if self.methodobj is not None:
            self.global_vars[self.objname] = self.methodobj

        argtypes, arg_mapping, constant_args = self._get_type_annotations(args, kwargs)

        # Add constant arguments to globals for caching
        self.global_vars.update(constant_args)

        # Cache key
        cachekey = self._cache.make_key(argtypes, self.closure_array_keys, self.closure_constant_keys, constant_args)

        if self._cache.has(cachekey):
            entry = self._cache.get(cachekey)
            # If the cache does not just contain a parsed SDFG
            if entry.compiled_sdfg is not None:
                kwargs.update(arg_mapping)
                entry.compiled_sdfg.clear_return_values()
                return entry.compiled_sdfg(**self._create_sdfg_args(entry.sdfg, args, kwargs))

        # Clear cache to enforce deletion and closure of compiled program
        # self._cache.pop()

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
        binaryobj = sdfg.compile(validate=self.validate)

        # Recreate key and add to cache
        cachekey = self._cache.make_key(argtypes, self.closure_array_keys, self.closure_constant_keys, constant_args)
        self._cache.add(cachekey, sdfg, binaryobj)

        # Call SDFG
        result = binaryobj(**sdfg_args)

        return result

    def _parse(self, args, kwargs, coarsen=None, save=False, validate=False) -> SDFG:
        """ 
        Try to parse a DaceProgram object and return the `dace.SDFG` object
        that corresponds to it.
        :param function: DaceProgram object (obtained from the ``@dace.program``
                        decorator).
        :param args: The given arguments to the function.
        :param kwargs: The given keyword arguments to the function.
        :param coarsen: Whether to apply dataflow coarsening or not (None
                       uses configuration-defined value). 
        :param save: If True, saves the generated SDFG to 
                    ``_dacegraphs/program.sdfg`` after parsing.
        :param validate: If True, validates the resulting SDFG after creation.
        :return: The generated SDFG object.
        """
        # Avoid import loop
        from dace.sdfg.analysis import scalar_to_symbol as scal2sym
        from dace.transformation import helpers as xfh

        # Obtain DaCe program as SDFG
        sdfg, cached = self._generate_pdp(args, kwargs, coarsen=coarsen)

        # Apply dataflow coarsening automatically
        if not cached and (coarsen == True or
                           (coarsen is None and Config.get_bool('optimizer', 'automatic_dataflow_coarsening'))):

            # Promote scalars to symbols as necessary
            promoted = scal2sym.promote_scalars_to_symbols(sdfg)
            if Config.get_bool('debugprint') and len(promoted) > 0:
                print('Promoted scalars {%s} to symbols.' % ', '.join(p for p in sorted(promoted)))

            sdfg.coarsen_dataflow()

            # Split back edges with assignments and conditions to allow richer
            # control flow detection in code generation
            xfh.split_interstate_edges(sdfg)

        # Save the SDFG. Skip this step if running from a cached SDFG, as
        # it might overwrite the cached SDFG.
        if not cached and not Config.get_bool('compiler', 'use_cache') and save:
            sdfg.save(os.path.join('_dacegraphs', 'program.sdfg'))

        # Validate SDFG
        if validate:
            sdfg.validate()

        return sdfg

    def _get_type_annotations(self, given_args: Tuple[Any],
                              given_kwargs: Dict[str, Any]) -> Tuple[ArgTypes, Dict[str, Any], Dict[str, Any]]:
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
        given_kwargs = {k: v for k, v in given_kwargs.items() if k not in self.symbols}

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
                    raise SyntaxError('Cannot compile DaCe program with type-annotated '
                                      'variable-length (starred) arguments and no given '
                                      'parameters. Please compile the program with arguments, '
                                      'call it without annotations, or remove the starred '
                                      f'arguments (invalid argument name: "{aname}").')

                types.update({f'__arg{j}': create_datadescriptor(varg) for j, varg in enumerate(vargs)})
                arg_mapping.update({f'__arg{j}': varg for j, varg in enumerate(vargs)})
                gvar_mapping[aname] = tuple(f'__arg{j}' for j in range(len(vargs)))
                # Shift arg_ind to the end
                arg_ind = len(given_args)
            elif sig_arg.kind is sig_arg.VAR_KEYWORD:
                vargs = {k: create_datadescriptor(v) for k, v in given_kwargs.items() if k not in types}
                # If an annotation is given but the argument list is empty, fail
                if not _is_empty(ann) and len(vargs) == 0:
                    raise SyntaxError('Cannot compile DaCe program with type-annotated '
                                      'variable-length (starred) keyword arguments and no given '
                                      'parameters. Please compile the program with arguments, '
                                      'call it without annotations, or remove the starred '
                                      f'arguments (invalid argument name: "{aname}").')
                types.update({f'__kwarg_{k}': v for k, v in vargs.items()})
                arg_mapping.update({f'__kwarg_{k}': given_kwargs[k] for k in vargs.keys()})
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
                            raise SyntaxError('Not enough arguments given to program (missing '
                                              f'argument: "{aname}").')
                    else:
                        if curarg is None:
                            curarg = given_args[arg_ind]
                        arg_ind += 1
                elif sig_arg.kind is sig_arg.POSITIONAL_OR_KEYWORD:
                    if arg_ind >= nargs:
                        if aname not in given_kwargs:
                            if curarg is None and not _is_empty(sig_arg.default):
                                curarg = sig_arg.default
                            elif curarg is None:
                                raise SyntaxError('Not enough arguments given to program (missing '
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
                            raise SyntaxError('Not enough arguments given to program (missing '
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

    def _load_sdfg(self, path: str, *args, **kwargs):
        """
        (Internal API)
        Loads an external SDFG that will be used when the function is called.
        :param path: Path to SDFG file.
        :param args: Optional compile-time arguments.
        :param kwargs: Optional compile-time keyword arguments.
        :return: A 2-tuple of (SDFG, program cache key)
        """
        # Read SDFG
        if path is not None:
            sdfg = SDFG.from_file(path)
        else:
            sdfg = None

        # Perform preprocessing to obtain closure
        argtypes, _, constant_args = self._get_type_annotations(args, kwargs)

        # Remove None arguments and make into globals that can be folded
        removed_args = set()
        for k, v in argtypes.items():
            if v.dtype.type is None:
                removed_args.add(k)
        argtypes = {k: v for k, v in argtypes.items() if v.dtype.type is not None}

        closure = self.closure_resolver(constant_args)

        # Create new argument mapping from closure arrays
        arg_mapping = {k: v for k, (_, _, v, _) in closure.closure_arrays.items()}
        self.closure_arg_mapping = arg_mapping
        self.closure_array_keys = set(closure.closure_arrays.keys()) - removed_args
        self.closure_constant_keys = set(closure.closure_constants.keys()) - removed_args
        self.resolver = closure

        return sdfg, self._cache.make_key(argtypes, self.closure_array_keys, self.closure_constant_keys, constant_args)

    def load_sdfg(self, path: str, *args, **kwargs) -> None:
        """
        Loads an external SDFG that will be used when the function is called.
        :param path: Path to SDFG file.
        :param args: Optional compile-time arguments.
        :param kwargs: Optional compile-time keyword arguments.
        """
        sdfg, cachekey = self._load_sdfg(path, *args, **kwargs)

        # Update SDFG cache with the SDFG (without a compiled version)
        self._cache.add(cachekey, sdfg, None)

    def load_precompiled_sdfg(self, path: str, *args, **kwargs) -> None:
        """
        Loads an external compiled SDFG object that will be invoked when the 
        function is called.
        :param path: Path to SDFG build folder (e.g., ".dacecache/program").
                     Path has to include ``program.sdfg`` and the binary shared
                     object under the ``build`` folder.
        :param args: Optional compile-time arguments.
        :param kwargs: Optional compile-time keyword arguments.
        """
        from dace.sdfg import utils as sdutil  # Avoid import loop
        csdfg = sdutil.load_precompiled_sdfg(path)
        _, cachekey = self._load_sdfg(None, *args, **kwargs)

        # Update SDFG cache with the SDFG and compiled version
        self._cache.add(cachekey, csdfg.sdfg, csdfg)

    def _generate_pdp(self, args, kwargs, coarsen=None) -> SDFG:
        """ Generates the parsed AST representation of a DaCe program.
            :param args: The given arguments to the program.
            :param kwargs: The given keyword arguments to the program.
            :param coarsen: Whether to apply dataflow coarsening when parsing 
                           nested dace programs.
            :return: A 2-tuple of (parsed SDFG object, was the SDFG retrieved
                     from cache).
        """
        dace_func = self.f

        # If exist, obtain type annotations (for compilation)
        argtypes, _, gvars = self._get_type_annotations(args, kwargs)

        # Move "self" from an argument into the closure
        if self.methodobj is not None:
            self.global_vars[self.objname] = self.methodobj

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
        removed_args = set()
        for k, v in argtypes.items():
            if v.dtype.type is None:
                global_vars[k] = None
                removed_args.add(k)
        argtypes = {k: v for k, v in argtypes.items() if v.dtype.type is not None}

        # Set module aliases to point to their actual names
        modules = {k: v.__name__ for k, v in global_vars.items() if dtypes.ismodule(v)}
        modules['builtins'] = ''

        # Add symbols as globals with their actual names (sym_0 etc.)
        global_vars.update({v.name: v for _, v in global_vars.items() if isinstance(v, symbolic.symbol)})
        for argtype in argtypes.values():
            global_vars.update({v.name: v for v in argtype.free_symbols})

        # Add constant arguments to global_vars
        global_vars.update(gvars)

        # Parse AST to create the SDFG
        parsed_ast, closure = preprocessing.preprocess_dace_program(dace_func,
                                                                    argtypes,
                                                                    global_vars,
                                                                    modules,
                                                                    resolve_functions=self.resolve_functions)

        # Create new argument mapping from closure arrays
        arg_mapping = {k: v for k, (_, _, v, _) in closure.closure_arrays.items()}
        self.closure_arg_mapping = arg_mapping
        self.closure_array_keys = set(closure.closure_arrays.keys()) - removed_args
        self.closure_constant_keys = set(closure.closure_constants.keys()) - removed_args
        self.resolver = closure

        # If parsed SDFG is already cached, use it
        cachekey = self._cache.make_key(argtypes, self.closure_array_keys, self.closure_constant_keys, gvars)
        if self._cache.has(cachekey):
            sdfg = self._cache.get(cachekey).sdfg
            cached = True
        else:
            cached = False
            try:
                sdfg = newast.parse_dace_program(self.name, parsed_ast, argtypes, self.dec_kwargs, closure, coarsen=coarsen)
            except Exception:
                if Config.get_bool('frontend', 'verbose_errors'):
                    from dace.frontend.python import astutils
                    print('VERBOSE: Failed to parse the following program:')
                    print(astutils.unparse(parsed_ast.preprocessed_ast))
                raise

            # Set SDFG argument names, filtering out constants
            sdfg.arg_names = [a for a in self.argnames if a in argtypes]

            # TODO: Add to parsed SDFG cache

        return sdfg, cached
