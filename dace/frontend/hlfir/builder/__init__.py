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
    emit_declare_transient,
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
    emit_symbol_init,
    emit_while,
)
from dace.frontend.hlfir.builder.emit_tasklet import emit_scalar_assign, emit_tasklet

# Default bridge pass pipeline.  Order matters — see ``README.md``.
DEFAULT_PIPELINE = (
    # Lower fir.select_case → arith.cmp + cf.cond_br BEFORE inline-all.
    # The upstream ``mlir::inlineCall`` mishandles fir.select_case's
    # block-operand remap and segfaults when a callee containing one is
    # inlined.  Pre-lowering side-steps the inliner crash and produces
    # a plain CFG that lift-cf-to-scf turns back into nested scf.if for
    # the bridge to consume.
    "lower-fir-select-case,"
    "hlfir-inline-all,"
    # Erase element-scoped alias declares left by inlining scalar-arg
    # procedures (elemental subroutines, most commonly) — runs before
    # flatten-structs so the rewrite's designate chains are already
    # single-declare rooted.
    "hlfir-fold-element-aliases,"
    # Replace ``hlfir.associate`` of an ``hlfir.elemental`` (Flang's
    # copy-in temp for noncontiguous slice arguments) with an explicit
    # ``fir.alloca`` + gather DO loop.  After inline-all so the
    # surrounding callee dummy declare aliasing the temp resolves
    # through the materialised hlfir.declare.
    "hlfir-materialise-associates,"
    # Scatter sibling: rewrites ``hlfir.region_assign`` whose lhs region
    # carries an ``hlfir.elemental_addr`` (Fortran ``d(cols) = source``)
    # into an explicit DO loop of per-element scalar assigns.
    "hlfir-expand-region-assign,"
    # Drop private callee bodies once inlined — otherwise their
    # declares leak into extract_vars as stray scalars.
    "symbol-dce,"
    # Statically devirtualise resolvable ``fir.dispatch`` /
    # ``fir.select_type`` ops.  The bridge supports CLASS-as-
    # monomorphic-box only — surviving polymorphic ops are caught
    # by ``hlfir-reject-polymorphism`` immediately after.
    "fir-polymorphic-op,"
    "hlfir-reject-polymorphism,"
    # Collapse Fortran sequence-association adapters (caller passing a
    # scalar element of an array where the formal expects an
    # explicit-shape array) into an explicit section designate of the
    # parent.  Runs AFTER inline-all (so the inlined callee body's
    # declare-of-converted-ref is visible) and BEFORE flatten-structs
    # (so the section view feeds into the usual designate-rewrite).
    "hlfir-rewrite-sequence-association,"
    "hlfir-flatten-structs,"
    # Collapse Fortran ``ptr => target`` rebinds under the strict-no-
    # aliasing assumption: every read or write of the pointer becomes
    # an access to the rebind target's storage.  Runs AFTER
    # flatten-structs so a target like ``s%a`` has already been
    # rewritten to a flat ``s_a`` declare; we trace the rebind through
    # the resulting box+embox chain to that flat declare and rewrite
    # all pointer reads accordingly.  Emits a warning per rewrite to
    # surface the no-alias assumption (Fortran allows aliased pointer
    # access; relying on alias semantics is unsafe under this pass).
    "hlfir-rewrite-pointer-assigns,"
    "hlfir-propagate-shapes,"
    "hlfir-default-intent,"
    # Lift cf.br / cf.cond_br loops into scf.while so extract_ast can walk them.
    "lift-cf-to-scf,"
    # Constant propagation + fold + CSE after every HLFIR rewrite has
    # exposed as many constants as it will.
    "sccp,canonicalize,cse")

# Multi-file pipeline: flatten cross-file calls into the entry, drop
# the now-dead sibling definitions, fail fast on anything left
# unresolved, then run the usual HLFIR rewrite chain.  The
# ``hlfir-inline-all`` pass needs the per-dialect
# DialectInlinerInterface to be attached to the MLIRContext, which
# the bridge's constructor now does via ``mlir::func::
# registerInlinerExtension`` + ``fir::addFIRInlinerExtension``.
MULTI_FILE_PIPELINE = ("hlfir-inline-all,"
                       "hlfir-fold-element-aliases,"
                       "symbol-dce,"
                       "hlfir-verify-no-unresolved-calls,"
                       "hlfir-flatten-structs,"
                       "hlfir-propagate-shapes,"
                       "hlfir-default-intent,"
                       "lift-cf-to-scf,"
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

    def __init__(self, hlfir_path: str, pipeline: str = DEFAULT_PIPELINE, entry: str | None = None):
        """Parse HLFIR, run the pass pipeline, and classify variables.

        If ``entry`` is set, every other ``func.func`` in the module is
        made private before the pipeline runs so ``symbol-dce`` drops
        them after ``hlfir-inline-all`` has flattened their bodies in.
        Needed when the source contains a module-scope callee that would
        otherwise leak dummy-arg declares into ``extract_vars``.
        """
        self.module = hb.HLFIRModule()
        if not self.module.parse_file(hlfir_path):
            raise RuntimeError(f"Cannot parse {hlfir_path}")

        if entry is not None:
            self.module.set_entry_symbol(entry)

        # Run bridge passes BEFORE extracting variables so assumed-shape
        # dummies pick up real names and the rest of the rewrites have
        # settled.
        if pipeline:
            self.module.run_passes(pipeline)

        self._classify()

    @classmethod
    def from_files(cls, hlfir_paths, *, entry: str, pipeline: str = MULTI_FILE_PIPELINE) -> "SDFGBuilder":
        """Parse and merge several HLFIR files, keep ``entry`` as the only
        public function, verify every remaining call resolves, then run
        the rewrite chain.

        Use this when the entry subroutine and its dependencies live in
        separate ``.hlfir`` files — e.g. the ICON multi-module flow where
        each module compiles to its own HLFIR.  ``parse_files`` dedups
        by symbol name so shared external declarations don't conflict.

        Arguments:
            hlfir_paths: list of paths to HLFIR files.  The first file
                         becomes the base; the rest are merged in.
            entry:       mangled Flang symbol (``_QPkernel`` /
                         ``_QMmodPsub``) of the subroutine the SDFG
                         should represent.
            pipeline:    pass pipeline to run before extraction.
        """
        obj = cls.__new__(cls)
        obj.module = hb.HLFIRModule()
        if not obj.module.parse_files(list(hlfir_paths)):
            raise RuntimeError(f"Cannot parse one of {hlfir_paths}")
        obj.module.set_entry_symbol(entry)
        if pipeline:
            obj.module.run_passes(pipeline)
        remaining = obj.module.list_functions()
        if entry not in remaining:
            raise RuntimeError(f"entry '{entry}' dropped by pipeline; remaining: {remaining}")
        obj._classify()
        return obj

    def _classify(self):
        """Shared post-parse extraction: variables + AST + role split."""
        self.variables = self.module.get_variables()
        self.ast = self.module.get_ast()
        self.arrays = {v.fortran_name: v for v in self.variables if v.role == "array"}
        self.symbols = {v.fortran_name: v for v in self.variables if v.role == "symbol"}
        self.scalars = {v.fortran_name: v for v in self.variables if v.role == "scalar"}
        # Per-axis offset symbols: ``offset_<arr>_d<i>`` is the SDFG
        # symbol every memlet of array ``<arr>`` subtracts on dim ``i``.
        # Populated by ``add_descriptors`` from each VarInfo's
        # ``lower_bounds``.  Values are int (constant-folded by
        # ``sdfg.specialize``), str (substituted with another symbol
        # name), or ``None`` (unknown — symbol stays free, caller
        # passes it).
        self.offset_values: dict[str, int | str | None] = {}

    def build(self) -> SDFG:
        """Construct the SDFG, run the unconditional offset-symbol
        specialisation pass, and attach a frozen-signature snapshot.

        The snapshot is later verified by ``codegen.generate_code``
        before any C++ header gets emitted — any downstream
        transformation that drifts the argument list will then raise
        rather than silently invalidate a generated Fortran binding.
        """
        self._id_counter = 0
        sdfg = SDFG(sdfg_name(self))
        add_descriptors(self, sdfg)
        # Constant-pool (Flang's ``_QQro.<...>`` globals): for every
        # ``parameter``-attributed declare whose backing global carries
        # a dense init, register the data via ``sdfg.add_constant``.
        # That bakes the values into codegen so the kernel's reads land
        # on the right data instead of an uninitialised transient.  The
        # array descriptor stays in ``sdfg.arrays`` (created by
        # ``add_descriptors``); the constant table just attaches the
        # initial-value tuple to it.
        self._register_constants(sdfg)
        ctx = _Ctx(sdfg, self)
        self._emit(ctx, self.ast, sdfg)
        ctx.flush(self)
        # Always-on post-emit substitution.  ``offset_values`` carries
        # two flavours of mapping: int constants (``offset_d_d0 = 50``
        # for ``dimension(50:54)``) and symbol aliases (``offset_d_d0 =
        # "arrsize"`` for ``dimension(arrsize:arrsize+4)``).  They take
        # different paths because ``sdfg.specialize`` only handles
        # constants — feeding it a string would land on ``add_constant``
        # and downstream casting tries ``int64("arrsize")`` and
        # ValueError-s.
        const_offsets, alias_offsets = {}, {}
        for k, v in self.offset_values.items():
            if v is None:
                continue
            (alias_offsets if isinstance(v, str) else const_offsets)[k] = v
        if const_offsets:
            sdfg.specialize(const_offsets)
        # Symbol-to-symbol aliasing (``offset_d_d0 = arrsize``): rename
        # every reference and drop the now-redundant offset symbol from
        # the SDFG so its signature only carries ``arrsize`` as a free
        # symbol.
        for src, dst in alias_offsets.items():
            sdfg.replace(src, dst)
            if src in sdfg.symbols:
                sdfg.symbols.pop(src)
        # Post-gen cleanups (Stage 4b in dace/frontend/hlfir/README.md).
        # Run BEFORE the FrozenSignature snapshot so the snapshot
        # captures the post-cleanup signature (matters for the
        # downstream codegen drift check).
        self._run_post_gen_passes(sdfg)
        self._attach_frozen_signature(sdfg)
        return sdfg

    def _register_constants(self, sdfg: SDFG) -> None:
        """Attach Flang's constant-pool data to the SDFG.

        Every ``VarInfo`` with non-empty ``const_data`` represents a
        ``_QQro.<shape>x<dtype>.<counter>`` global — the read-only
        backing for an array or scalar literal in the source.  The
        bridge has already added a transient descriptor for it via
        ``add_descriptors``; this hook attaches the dense values so
        DaCe's codegen materialises them into the binary.

        The data widens to ``double`` on the bridge side for
        transport; we narrow to the descriptor's actual dtype here
        and reshape back to the rank-N companion shape.  Scalar
        constants (rank 0, single value) are uncommon — Fortran
        ``parameter`` scalars typically inline as ``arith.constant``
        — but the path supports them with a trivial 1-element array.
        """
        import numpy as np
        for v in self.variables:
            if not v.const_data:
                continue
            if v.fortran_name not in sdfg.arrays:
                continue
            desc = sdfg.arrays[v.fortran_name]
            np_dtype = desc.dtype.as_numpy_dtype()
            arr = np.asarray(v.const_data, dtype=np.float64).astype(np_dtype)
            if v.rank > 0:
                # Bridge transports row-major doubles; reshape to
                # the descriptor's declared shape.  The bridge
                # currently only emits constant pools for rank-1
                # globals (Flang shape: a flat dense<[...]> tensor),
                # so this is a no-op pass-through today, but the
                # reshape keeps the contract honest if higher-rank
                # globals start surfacing.
                shape = tuple(int(d) for d in desc.shape)
                if arr.size == int(np.prod(shape)):
                    arr = arr.reshape(shape, order='C')
            sdfg.add_constant(v.fortran_name, arr, desc)

    def _run_post_gen_passes(self, sdfg: SDFG) -> None:
        """Run the post-generation cleanup passes that take a freshly-
        emitted bridge SDFG to its canonical shape.  See Stage 4b in
        ``dace/frontend/hlfir/README.md`` for the pipeline.

        Currently:
            * ``SSALoopIterators`` -- rewrites every ``LoopRegion``'s
              loop variable to a globally-unique ``_it_<N>`` symbol and
              propagates the rename through the body.
            * ``replace_length_one_arrays_with_scalars`` -- folds
              every length-1 ``Array`` on the SDFG signature down to a
              true ``Scalar``.  The bridge already emits scalar inputs
              directly as ``Scalar``; this pass cleans up leftover
              length-1 OUTPUTS and any local 1-element transients so
              callers can bind plain ``int`` / ``float`` instead of
              wrapping in a numpy 1-array.
        """
        from dace.sdfg.construction_utils import replace_length_one_arrays_with_scalars
        from dace.transformation.passes.ssa_loop_iterators import SSALoopIterators

        # Empty-region cleanup: any ControlFlowRegion (LoopRegion,
        # ConditionalBlock branch, the top-level SDFG, etc.) that
        # ended up with zero internal blocks gets a single empty
        # state added.  Validation requires every CFG region to
        # have a defined start block; an empty region triggers
        # "Ambiguous starting block".  Such empties arise legitimately
        # from Fortran source — a ``do i = 1, N; <only-stripped-by-
        # flatten>; end do`` whose body became a no-op after
        # AoS+allocatable flattening, an empty IF branch, etc.
        # The empty state is semantically equivalent to the source
        # construct (a loop iterating over a no-op body is still
        # a no-op overall).
        for region in list(sdfg.all_control_flow_regions()):
            if len(list(region.nodes())) == 0:
                region.add_state("empty_body", is_start_block=True)

        SSALoopIterators().apply_pass(sdfg, {})
        # ``transient_only=True``: only fold LOCAL 1-element transients
        # (e.g. accumulators left as length-1 arrays by the bridge).  The
        # signature convention is preserved: ``intent(out)`` / ``inout``
        # scalars stay as length-1 ``Array`` so callers can pass a numpy
        # 1-element buffer to receive the value.  ``intent(in)`` /
        # ``VALUE`` scalars are already emitted as ``Scalar`` directly by
        # ``descriptors.py`` and don't need this pass.
        replace_length_one_arrays_with_scalars(sdfg, recursive=True, transient_only=True)

    def _attach_frozen_signature(self, sdfg: SDFG) -> None:
        """Snapshot ``sdfg.arglist()`` + free symbols into a
        ``FrozenSignature`` and pin it on the SDFG.

        ``kind`` is read off the live SDFG descriptor, not the builder's
        role split: a scalar OUTPUT (``intent(out)`` / ``intent(inout)``)
        registers in ``self.scalars`` but lives on the SDFG as a length-1
        ``Array`` -- the bindings emitter must see ``kind='array'`` so it
        emits ``type(c_ptr), value`` (pointer) instead of a pass-by-value
        scalar binding.  A scalar INPUT (``intent(in)`` / ``VALUE``) lives
        as a true ``Scalar`` and gets ``kind='scalar'``.
        """
        # Local import keeps the binding machinery optional -- plain
        # ``import dace.frontend.hlfir`` doesn't drag it in.
        from dace.data import Array as _Array, Scalar as _Scalar
        from dace.frontend.hlfir.bindings.frozen_signature import FrozenArg, FrozenSignature

        args_list = []
        for sdfg_name_, desc in sdfg.arglist().items():
            v = (self.arrays.get(sdfg_name_) or self.symbols.get(sdfg_name_) or self.scalars.get(sdfg_name_))
            if sdfg_name_ in self.symbols:
                kind = 'symbol'
            elif isinstance(desc, _Scalar):
                kind = 'scalar'
            elif isinstance(desc, _Array):
                kind = 'array'
            else:
                kind = 'scalar'
            dtype_obj = getattr(desc, 'dtype', None)
            dtype_str = (getattr(dtype_obj, 'to_string', lambda: str(dtype_obj))() if dtype_obj is not None else '?')
            shape = tuple(str(s) for s in getattr(desc, 'shape', ()))
            args_list.append(
                FrozenArg(
                    fortran_name=v.fortran_name if v is not None else sdfg_name_,
                    sdfg_name=sdfg_name_,
                    kind=kind,
                    dtype=dtype_str,
                    rank=len(shape) if kind == 'array' else 0,
                    shape=shape,
                    intent=(v.intent if v is not None else ''),
                ))
        fs = FrozenSignature(
            entry=sdfg.name,
            mangled=next((v.mangled_name for v in self.arrays.values() if getattr(v, 'mangled_name', '')), sdfg.name),
            args=tuple(args_list),
            free_symbols=tuple(sorted(str(s) for s in sdfg.free_symbols)),
        )
        sdfg._frozen_signature = fs

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
        "declare_transient": emit_declare_transient,
        "symbol_init": emit_symbol_init,
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


def generate_sdfg(path: str = None, *, pipeline: str = None, entry: str = None, hlfir_files=None) -> SDFG:
    """Build an SDFG from one or several HLFIR files.

    Single-file form (back-compat):
        ``generate_sdfg("code.hlfir")`` — parses + DEFAULT_PIPELINE.

    Multi-file form (ICON-style linked entry):
        ``generate_sdfg(entry="_QPkernel", hlfir_files=[...])`` — parses
        every file, merges them, drops non-entry siblings, errors on
        unresolved calls, then runs the HLFIR rewrite chain.
    """
    if hlfir_files is not None:
        if entry is None:
            raise ValueError("entry= is required when hlfir_files= is supplied")
        return SDFGBuilder.from_files(
            hlfir_files,
            entry=entry,
            pipeline=(pipeline if pipeline is not None else MULTI_FILE_PIPELINE),
        ).build()
    if path is None:
        raise TypeError("generate_sdfg: pass a path or hlfir_files=[...]")
    return SDFGBuilder(
        path,
        pipeline=(pipeline if pipeline is not None else DEFAULT_PIPELINE),
    ).build()
