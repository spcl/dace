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
from dace.cli.progress import OptionalProgressBar
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import preprocessing
from dace.frontend.python.nextgen.canonical.passes import default_passes
from dace.frontend.python.nextgen.common import (CanonicalViolationError, FrontendError, TreeVerificationError,
                                                 UnsupportedFeatureError)
from dace.frontend.python.nextgen.lowering.emitter import TreeEmitter
from dace.frontend.python.nextgen.lowering.mechanisms.return_elision import elide_return_copies
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
                        callbacks: Optional[Dict[str, Any]] = None,
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
    :param callbacks: Python callables of the callbacks preprocessing detected,
                      keyed by the same names as ``callback_mapping``. Needed
                      by statements that end up executing in the interpreter,
                      which reference the callable by that name.
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
    context.callback_callables.update(callbacks or {})
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
                               folded_constants=context.folded_constants,
                               callback_mapping=dict(callback_mapping or {}),
                               arg_names=list(arg_names or argtypes.keys()))

    # Stage 3: rule-driven lowering through the closed emitter. Progress
    # feedback appears only for lengthy parses (threshold- and config-gated);
    # the statement count is a lower bound (inlined callees also tick).
    state = LoweringState(context, TreeEmitter(root))
    total_statements = sum(1 for node in ast.walk(program) if isinstance(node, ast.stmt))
    state.progress = OptionalProgressBar(n=total_statements, title=f'Parsing {name}')
    try:
        state.lower_body(program.body)
    finally:
        state.progress.done()

    # A returned program-local array can carry the return container's name
    # instead of being copied into it. Done on the finished tree, where every
    # use of the container is visible.
    elide_return_copies(root)

    # Stage 4: verification of the output contract
    verify_tree(root)

    # The tree carries the source of every callback it emitted; turning that
    # into live callables here is what makes the tree (and the SDFG converted
    # from it) callable without the caller reconstructing them.
    root.materialize_callbacks()
    return root


def parse_program(program, *args, debug: bool = False, **kwargs) -> tn.ScheduleTreeRoot:
    """
    Convenience entry point: preprocess a :class:`DaceProgram` and build a
    verified schedule tree from it.

    :param program: The ``@dace.program`` to lower.
    :param args: JIT argument examples.
    :param debug: If True, runs extra verification between pipeline passes.
    :param kwargs: JIT keyword argument examples.
    :return: A verified :class:`ScheduleTreeRoot`.
    """
    return program.to_schedule_tree(*args, debug=debug, **kwargs)


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
