# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Next-generation Python frontend: lowers preprocessed ``@dace.program`` ASTs to
verified schedule trees through a staged pipeline (canonicalization, semantic
binding, rule-driven lowering, verification).

See :mod:`dace.frontend.python.nextgen.pipeline` for the stage contracts.
"""
import ast
from typing import Any, Dict, Optional, Sequence, Tuple

from dace import data
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import preprocessing
from dace.frontend.python.nextgen.canonical.passes import default_passes
from dace.frontend.python.nextgen.common import (CanonicalViolationError, FrontendError, TreeVerificationError,
                                                 UnsupportedFeatureError)
from dace.frontend.python.nextgen.lowering.emitter import TreeEmitter
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

    for argument_name, descriptor in list(argtypes.items()):
        if isinstance(descriptor, data.View):
            argtypes[argument_name] = descriptor.as_array()
        descriptor = argtypes[argument_name]
        descriptor.transient = False

    global_vars = dict(program.global_vars)
    unspecified_defaults = {key: value for key, value in program.default_args.items() if key not in specified}
    gvars.update(unspecified_defaults)
    global_vars.update(gvars)
    modules = {
        key: value.__name__
        for key, value in global_vars.items() if hasattr(value, '__name__') and type(value).__name__ == 'module'
    }
    modules['builtins'] = ''

    parsed_ast, closure = preprocessing.preprocess_dace_program(program.f,
                                                                argtypes,
                                                                global_vars,
                                                                modules,
                                                                resolve_functions=program.resolve_functions,
                                                                default_args=unspecified_defaults.keys())

    constants: Dict[str, Tuple[data.Data, Any]] = {}
    for constant_name, value in closure.closure_constants.items():
        try:
            descriptor = data.create_datadescriptor(value)
        except (TypeError, ValueError):
            continue
        constants[constant_name] = (descriptor, value)

    callback_mapping = {key: original for key, (original, _, _) in closure.callbacks.items()}
    arg_names = [argument_name for argument_name in program.argnames if argument_name in argtypes]

    return build_schedule_tree(program.name,
                               parsed_ast,
                               argtypes,
                               constants=constants,
                               callback_mapping=callback_mapping,
                               arg_names=arg_names,
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
