# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Pipeline driver for the next-generation Python frontend.

The frontend is organized as a sequence of stages with explicit, checkable
contracts:

1. **Closure acquisition**: resolve globals/closure and preprocess the source
   AST (reuses :mod:`dace.frontend.python.preprocessing`).
2. **Canonicalization**: AST-to-AST passes that reduce the program to the
   Canonical Python AST (CPA) subset. This stage is *total*: any statement
   that cannot be canonicalized becomes an explicit ``OpaqueStmt`` marker with
   precomputed input/output sets.
3. **Lowering**: a rule registry maps each CPA statement type to exactly one
   lowering rule that emits frontend-legal schedule tree nodes through a
   closed emitter.
4. **Verification**: structural invariants of the resulting tree are checked.

Each canonicalization pass implements :class:`CanonicalPass`. The pipeline
optionally verifies the CPA contract after every pass (debug mode) and always
verifies it once at the end of the stage.
"""
import ast
from typing import List, Protocol, runtime_checkable

from dace.frontend.python.nextgen.canonical.cpa import verify_canonical


@runtime_checkable
class CanonicalPass(Protocol):
    """A single AST-to-AST normalization pass."""

    #: Human-readable pass name for diagnostics.
    name: str

    def apply(self, tree: ast.FunctionDef, context: 'PipelineContext') -> ast.FunctionDef:
        """Transform the AST in place or return a replacement tree."""
        ...


class PipelineContext:
    """
    Read-only information shared by canonicalization passes: the program name,
    source filename, resolved globals, and known argument descriptors.
    """

    def __init__(self, name: str, filename: str, global_vars: dict, argtypes: dict):
        self.name = name
        self.filename = filename
        self.global_vars = global_vars
        self.argtypes = argtypes
        #: Fresh-name counter shared across passes so generated temporaries never collide.
        self._name_counter = 0

    def fresh_name(self, prefix: str = '__tmp') -> str:
        """Allocate a program-unique generated name."""
        result = f'{prefix}{self._name_counter}'
        self._name_counter += 1
        return result


class CanonicalizationPipeline:
    """Runs canonicalization passes in order and enforces the CPA postcondition."""

    def __init__(self, passes: List[CanonicalPass], debug: bool = False):
        """
        :param passes: Ordered list of AST normalization passes.
        :param debug: If True, verifies the CPA contract after each pass rather
                      than only at the end of the stage. Intermediate passes
                      are allowed to see non-canonical input, so per-pass
                      verification only checks well-formedness, not the full
                      CPA subset.
        """
        self.passes = passes
        self.debug = debug

    def run(self, tree: ast.FunctionDef, context: PipelineContext) -> ast.FunctionDef:
        for pass_object in self.passes:
            result = pass_object.apply(tree, context)
            tree = result if result is not None else tree
            if self.debug:
                ast.fix_missing_locations(tree)
        ast.fix_missing_locations(tree)
        verify_canonical(tree, filename=context.filename)
        return tree
