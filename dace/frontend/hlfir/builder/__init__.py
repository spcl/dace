"""``SDFGBuilder`` — walks the HLFIR AST to directly construct a DaCe SDFG.

Pipeline:
    flang-20 -fc1 -emit-hlfir code.f90 -o code.hlfir
    sdfg = generate_sdfg("code.hlfir")   # → dace.SDFG, validated

Architecture:
    The builder parses the HLFIR via the C++ bridge (``hlfir_bridge.so``),
    runs the default pass pipeline, then walks the recursive ASTNode tree
    and emits DaCe constructs:

        kind="assign"      → Tasklet / interstate-edge assignment
        kind="loop"        → LoopRegion (nested for-loop)
        kind="while"       → LoopRegion (while form, post lift-cf-to-scf)
        kind="conditional" → ConditionalBlock + ControlFlowRegion per branch
        kind="copy"        → CopyLibraryNode
        kind="memset"      → MemsetLibraryNode
        kind="libcall"     → BLAS / standard library node (MatMul, Dot, …)
        kind="reduce"      → standard.Reduce
        kind="break"       → BreakBlock
        kind="return"      → ReturnBlock

Per-emitter implementations live in sibling modules under this package.
``SDFGBuilder`` itself keeps only orchestration — ``__init__``, ``build``,
``nid``, and the ``_emit`` dispatch.

State-change rules:
    - Write to a symbol → interstate edge with assignment (emit_assign).
    - Every other write → tasklet in the current state.
    - LoopRegion / ConditionalBlock open a fresh region; their children
      run in a nested ``_Ctx``.

NOTE on nanobind bindings:
    Every read of a std::vector-typed attribute (e.g. ast_node.children,
    var.shape_symbols, assign.accesses) returns a FRESH Python list copy.
    Hot paths cache such attributes into locals.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

from dace import SDFG

from build_bridge import hb

# intrinsics/ lives alongside this file but uses absolute imports
# (``dace.frontend.hlfir.intrinsics``).  hlfir_to_sdfg is usually imported
# via the ``sys.path.insert(<hlfir_dir>, …)`` pattern in build_bridge.py,
# which does not expose the full ``dace.frontend.hlfir`` package.  Add the
# DaCe source root so the package import resolves either way.
_BUILDER_DIR = _Path(__file__).resolve().parent
# parents: builder/ → hlfir/ → frontend/ → dace/ → <repo root>.
_REPO_ROOT = _BUILDER_DIR.parents[3]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from dace.frontend.hlfir.builder.context import _Ctx
from dace.frontend.hlfir.builder.descriptors import (
    DTYPE,
    add_descriptors,
    auto_declare_synth,
    dt,
    sdfg_name,
)
from dace.frontend.hlfir.builder.emit_library import (
    emit_break,
    emit_copy,
    emit_libcall,
    emit_memset,
    emit_reduce,
    emit_return,
)
from dace.frontend.hlfir.builder.emit_cfg import (
    emit_assign,
    emit_cond,
    emit_loop,
    emit_while,
)
from dace.frontend.hlfir.builder.emit_tasklet import emit_scalar_assign, emit_tasklet

# Default bridge pass pipeline.  Order matters — see ``README.md``.
DEFAULT_PIPELINE = (
    "hlfir-inline-all,"
    "hlfir-flatten-structs,"
    "hlfir-propagate-shapes,"
    "hlfir-default-intent,"
    # Lift cf.br / cf.cond_br loops into scf.while so extract_ast can walk them.
    "lift-cf-to-scf,"
    # Constant propagation + fold + CSE after every HLFIR rewrite has
    # exposed as many constants as it will.
    "sccp,canonicalize,cse")


class SDFGBuilder:
    """Walks the HLFIR ASTNode tree and emits a DaCe SDFG.

    Public surface:
        builder = SDFGBuilder("code.hlfir")
        sdfg    = builder.build()

    After construction:
        self.variables  — full VarInfo list from the bridge.
        self.arrays     — {name: VarInfo} for rank>0 variables.
        self.symbols    — {name: VarInfo} for scalars used in shapes / bounds /
                          control-flow conditions (pass 2b–2d of extract_vars).
        self.scalars    — {name: VarInfo} for remaining scalars.
    """

    DTYPE = DTYPE

    def __init__(self, hlfir_path: str, pipeline: str = DEFAULT_PIPELINE):
        """Parse HLFIR, run the pass pipeline, and classify variables."""
        self.module = hb.HLFIRModule()
        if not self.module.parse_file(hlfir_path):
            raise RuntimeError(f"Cannot parse {hlfir_path}")

        # Run bridge passes BEFORE extracting variables so assumed-shape
        # dummies pick up real names and the rest of the rewrites have
        # settled.
        if pipeline:
            self.module.run_passes(pipeline)

        self.variables = self.module.get_variables()
        self.ast = self.module.get_ast()
        self.arrays = {v.fortran_name: v for v in self.variables if v.role == "array"}
        self.symbols = {v.fortran_name: v for v in self.variables if v.role == "symbol"}
        self.scalars = {v.fortran_name: v for v in self.variables if v.role == "scalar"}

    def build(self) -> SDFG:
        """Construct and return a validated SDFG."""
        self._id_counter = 0
        sdfg = SDFG(sdfg_name(self))
        add_descriptors(self, sdfg)
        ctx = _Ctx(sdfg, self)
        self._emit(ctx, self.ast, sdfg)
        ctx.flush(self)
        return sdfg

    def nid(self) -> int:
        """Globally unique integer.  Shared across ``_Ctx`` instances so
        loop variable names (``jk_0``, ``jc_1``, ``jk_2``, …) never
        collide.
        """
        i = self._id_counter
        self._id_counter += 1
        return i

    _EMIT_DISPATCH = {
        "assign": emit_assign,
        "loop": emit_loop,
        "while": emit_while,
        "conditional": emit_cond,
        "reduce": emit_reduce,
        "copy": emit_copy,
        "memset": emit_memset,
        "libcall": emit_libcall,
        "break": emit_break,
        "return": emit_return,
    }

    def _emit(self, ctx: '_Ctx', nodes: list, region):
        """Recursive dispatcher — maps each ASTNode.kind to its emitter."""
        for n in nodes:
            fn = self._EMIT_DISPATCH.get(n.kind)
            if fn is not None:
                fn(self, ctx, n, region)
            # "call" currently has no emitter — nested SDFG / library node
            # is a future feature (Phase 4).

    # Scalar-assign is called from _Ctx.flush; keep it as a method on the
    # builder for that caller's convenience.
    def emit_scalar_assign(self, state, target: str, value: str):
        emit_scalar_assign(self, state, target, value)


def generate_sdfg(path: str, pipeline: str = DEFAULT_PIPELINE) -> SDFG:
    """One-liner: parse HLFIR file → run passes → validated DaCe SDFG."""
    return SDFGBuilder(path, pipeline=pipeline).build()
