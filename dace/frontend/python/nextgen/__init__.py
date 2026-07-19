# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Next-generation Python frontend: lowers preprocessed ``@dace.program`` ASTs to
verified schedule trees through a staged pipeline (canonicalization, semantic
binding, rule-driven lowering, verification).

See :mod:`dace.frontend.python.nextgen.pipeline` for the stage contracts.
"""
import ast
import copy
from typing import Any, Dict, Optional, Sequence, Tuple

from dace import data, dtypes, symbolic
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import preprocessing
from dace.frontend.python.nextgen.canonical.passes import default_passes
from dace.frontend.python.nextgen.common import (CanonicalViolationError, FrontendError, TreeVerificationError,
                                                 UnsupportedFeatureError)
from dace.frontend.python.nextgen.lowering.emitter import TreeEmitter
from dace.frontend.python.nextgen.lowering.parse_cache import warm_nested_parses
from dace.frontend.python.nextgen.lowering.registry import LoweringState
from dace.frontend.python.nextgen.pipeline import CanonicalizationPipeline, PipelineContext
from dace.frontend.python.nextgen.semantics.context import ProgramContext
from dace.frontend.python.nextgen.verify import verify_tree

# Importing the rule modules registers all lowering rules.
from dace.frontend.python.nextgen.lowering.rules import (assign, callbacks, calls, control_flow, dataflow_explicit,
                                                         returns)  # noqa: F401


def build_schedule_tree(name: str,
                        parsed_ast: preprocessing.PreprocessedAST,
                        argtypes: Dict[str, data.Data],
                        *,
                        constants: Optional[Dict[str, Tuple[data.Data, Any]]] = None,
                        callback_mapping: Optional[Dict[str, str]] = None,
                        arg_names: Optional[Sequence[str]] = None,
                        closure_arrays: Optional[Dict[str, Tuple[str, data.Data]]] = None,
                        debug: bool = False) -> tn.ScheduleTreeRoot:
    """
    Build a verified schedule tree from a preprocessed Python program AST.

    :param name: Program name.
    :param parsed_ast: Preprocessed program AST and metadata.
    :param argtypes: Mapping from argument names to data descriptors. The
                     descriptors are registered in the resulting tree's
                     repository by reference (not cloned).
    :param constants: Compile-time constants as (descriptor, value) tuples.
    :param callback_mapping: Mapping from callback symbol names to original
                             function names.
    :param arg_names: Ordered argument names.
    :param closure_arrays: External arrays referenced by the program, as a
                           mapping from the preprocessed reference name to
                           (source qualified name, descriptor). Registered as
                           non-transient containers.
    :param debug: If True, runs extra verification between pipeline passes.
    :return: A verified :class:`ScheduleTreeRoot`.
    """
    program = _program_node(parsed_ast)

    # Stage 1: canonicalization (total; ends with a verified CPA contract)
    pipeline_context = PipelineContext(name, parsed_ast.filename, parsed_ast.program_globals, argtypes)
    pipeline = CanonicalizationPipeline(default_passes(), debug=debug)
    program = pipeline.run(program, pipeline_context)

    # Stage 2: semantic context (single repository, shared with the tree root)
    context = ProgramContext(name, parsed_ast.filename, argtypes, parsed_ast.program_globals, constants or {})
    for reference_name, (qualified_name, descriptor) in (closure_arrays or {}).items():
        container = context.register_closure_array(reference_name, qualified_name, descriptor)
        context.bind(reference_name, container)

    # Stage 2.5: speculatively pre-parse nested @dace.program callees in
    # parallel (bottom-up), so sequential lowering hits the parse cache.
    warm_nested_parses(program.body, context)

    root = tn.ScheduleTreeRoot(name=name,
                               children=[],
                               containers=context.containers,
                               symbols=context.symbols,
                               constants=context.constants,
                               callback_mapping=dict(callback_mapping or {}),
                               arg_names=list(arg_names or argtypes.keys()))

    # Stage 3: rule-driven lowering through the closed emitter
    state = LoweringState(context, TreeEmitter(root))
    state.lower_body(program.body)

    # Stage 4: verification of the output contract
    verify_tree(root)
    return root


def parse_program(program, *args, debug: bool = False, **kwargs) -> tn.ScheduleTreeRoot:
    """
    Convenience entry point: preprocess a :class:`DaceProgram` and build a
    verified schedule tree from it.

    This mirrors the argument-type resolution of
    ``DaceProgram._generate_schedule_tree`` in compact form and exists for
    tests and direct use until the parser switches over to this pipeline.
    """
    argtypes, _, gvars, specified = program._get_type_annotations(args, kwargs)

    # Copy argument descriptors before marking them non-transient: annotation
    # descriptors are shared across calls and must not be mutated.
    for argument_name, descriptor in list(argtypes.items()):
        if isinstance(descriptor, data.View):
            descriptor = descriptor.as_array()
        else:
            descriptor = copy.deepcopy(descriptor)
        descriptor.transient = False
        argtypes[argument_name] = descriptor

    global_vars = dict(program.global_vars)

    # Bound methods: "self" is resolved through the closure, not an argument
    if program.methodobj is not None and program.objname is not None:
        global_vars[program.objname] = program.methodobj

    # None-valued arguments become foldable globals instead of containers
    removed_args = set()
    for argument_name, descriptor in argtypes.items():
        if descriptor.dtype.type is None:
            global_vars[argument_name] = None
            removed_args.add(argument_name)

    modules = {key: value.__name__ for key, value in global_vars.items() if dtypes.ismodule(value)}
    modules['builtins'] = ''

    # Symbols also resolve under their actual names (aliased symbol globals)
    global_vars.update(
        {value.name: value
         for value in list(global_vars.values()) if isinstance(value, symbolic.symbol)})

    unspecified_defaults = {key: value for key, value in program.default_args.items() if key not in specified}
    removed_args.update(unspecified_defaults)
    gvars.update(unspecified_defaults)
    global_vars.update(gvars)

    argtypes = {key: value for key, value in argtypes.items() if key not in removed_args}
    for descriptor in argtypes.values():
        global_vars.update({free_symbol.name: free_symbol for free_symbol in descriptor.free_symbols})

    parsed_ast, closure = preprocessing.preprocess_dace_program(program.f,
                                                                argtypes,
                                                                global_vars,
                                                                modules,
                                                                resolve_functions=program.resolve_functions,
                                                                default_args=unspecified_defaults.keys())

    constants: Dict[str, Tuple[data.Data, Any]] = {}
    for constant_name, value in closure.closure_constants.items():
        if constant_name in removed_args:
            continue
        try:
            descriptor = data.create_datadescriptor(value)
        except (TypeError, ValueError):
            continue
        constants[constant_name] = (descriptor, value)

    callback_mapping = {key: original for key, (original, _, _) in closure.callbacks.items()}
    arg_names = [argument_name for argument_name in program.argnames if argument_name in argtypes]
    closure_arrays = {
        reference_name: (qualified_name, descriptor)
        for reference_name, (qualified_name, descriptor, _, _) in closure.closure_arrays.items()
        if reference_name not in removed_args
    }

    return build_schedule_tree(program.name,
                               parsed_ast,
                               argtypes,
                               constants=constants,
                               callback_mapping=callback_mapping,
                               arg_names=arg_names,
                               closure_arrays=closure_arrays,
                               debug=debug)


def _program_node(parsed_ast: preprocessing.PreprocessedAST) -> ast.FunctionDef:
    program_ast = parsed_ast.preprocessed_ast
    node = program_ast.body[0] if isinstance(program_ast, ast.Module) else program_ast
    if not isinstance(node, ast.FunctionDef):
        raise FrontendError('Expected a preprocessed FunctionDef as frontend input', parsed_ast.filename)
    return node


__all__ = [
    'build_schedule_tree',
    'parse_program',
    'FrontendError',
    'UnsupportedFeatureError',
    'CanonicalViolationError',
    'TreeVerificationError',
]
