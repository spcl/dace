"""``SDFGBuilder``  --  walks the HLFIR AST to directly construct a DaCe SDFG.

Pipeline:
    flang-20 -fc1 -emit-hlfir code.f90 -o code.hlfir
    sdfg = generate_sdfg("code.hlfir")   # -> dace.SDFG, validated

Architecture:
    The builder parses the HLFIR via the C++ bridge (``hlfir_bridge.so``),
    runs the default pass pipeline, then walks the recursive ASTNode tree
    and emits DaCe constructs:

        kind="assign"      -> Tasklet / interstate-edge assignment
        kind="loop"        -> LoopRegion (nested for-loop)
        kind="while"       -> LoopRegion (while form, post lift-cf-to-scf)
        kind="conditional" -> ConditionalBlock + ControlFlowRegion per branch
        kind="copy"        -> CopyLibraryNode
        kind="memset"      -> MemsetLibraryNode
        kind="libcall"     -> BLAS / standard library node (MatMul, Dot, ...)
        kind="reduce"      -> standard.Reduce
        kind="break"       -> BreakBlock
        kind="return"      -> ReturnBlock

Per-emitter implementations live in sibling modules under this package.
``SDFGBuilder`` itself keeps only orchestration  --  ``__init__``, ``build``,
``nid``, and the ``_emit`` dispatch.

State-change rules:
    - Write to a symbol -> interstate edge with assignment (emit_assign).
    - Every other write -> tasklet in the current state.
    - LoopRegion / ConditionalBlock open a fresh region; their children
      run in a nested ``_Ctx``.

NOTE on nanobind bindings:
    Every read of a std::vector-typed attribute (e.g. ast_node.children,
    var.shape_symbols, assign.accesses) returns a FRESH Python list copy.
    Hot paths cache such attributes into locals.
"""

import sys as _sys
from pathlib import Path as _Path

from dace import InterstateEdge, SDFG

from build_bridge import hb

# intrinsics/ lives alongside this file but uses absolute imports
# (``dace.frontend.hlfir.intrinsics``).  hlfir_to_sdfg is usually imported
# via the ``sys.path.insert(<hlfir_dir>, ...)`` pattern in build_bridge.py,
# which does not expose the full ``dace.frontend.hlfir`` package.  Add the
# DaCe source root so the package import resolves either way.
_BUILDER_DIR = _Path(__file__).resolve().parent
# parents: builder/ -> hlfir/ -> frontend/ -> dace/ -> <repo root>.
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

# Default bridge pass pipeline.  Order matters  --  see ``README.md``.
DEFAULT_PIPELINE = (
    # Lower fir.select_case -> arith.cmp + cf.cond_br BEFORE inline-all.
    # The upstream ``mlir::inlineCall`` mishandles fir.select_case's
    # block-operand remap and segfaults when a callee containing one is
    # inlined.  Pre-lowering side-steps the inliner crash and produces
    # a plain CFG that lift-cf-to-scf turns back into nested scf.if for
    # the bridge to consume.
    "lower-fir-select-case,"
    "hlfir-inline-all,"
    # Erase element-scoped alias declares left by inlining scalar-arg
    # procedures (elemental subroutines, most commonly)  --  runs before
    # flatten-structs so the rewrite's designate chains are already
    # single-declare rooted.
    "hlfir-fold-element-aliases,"
    # Replace ``hlfir.associate`` of an ``hlfir.elemental`` (Flang's
    # copy-in temp for noncontiguous slice arguments) with an explicit
    # ``fir.alloca`` + gather DO loop.  After inline-all so the
    # surrounding callee dummy declare aliasing the temp resolves
    # through the materialised hlfir.declare.
    "hlfir-expand-vector-subscript-gather,"
    # Scatter sibling: rewrites ``hlfir.region_assign`` whose lhs region
    # carries an ``hlfir.elemental_addr`` (Fortran ``d(cols) = source``)
    # into an explicit DO loop of per-element scalar assigns.
    "hlfir-expand-vector-subscript-scatter,"
    # Drop private callee bodies once inlined  --  otherwise their
    # declares leak into extract_vars as stray scalars.
    "symbol-dce,"
    # Statically devirtualise resolvable ``fir.dispatch`` /
    # ``fir.select_type`` ops.  The bridge supports CLASS-as-
    # monomorphic-box only  --  surviving polymorphic ops are caught
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
    # Lift ``type(t), allocatable :: f(:)`` struct members (alloc-array
    # of records  --  ICON's ``p_patch%pprog(jg)`` shape) into top-level
    # companions with a leading runtime-extent dim.  Runs BEFORE
    # flatten-structs so the outer struct sees clean top-level arrays
    # after the lift.  Bails silently when no such member is present;
    # FlattenStructs's opaque-skip for alloc-array-of-records members
    # provides the safety net for un-handled patterns.
    "hlfir-lift-alloc-array-of-records,"
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
    # Lift array-reducing intrinsics (sum/maxval/minval/product/any/all)
    # that appear as INLINE expression operands into a preceding
    # scalar-temp assign.  ``buildExpr`` can't render reductions in a
    # tasklet expression  --  ``out = max(x, MAXVAL(slice))`` would otherwise
    # surface as ``_out = max(_in_x, ?)`` and crash Python ast.parse.
    # After this pass, the lifted ``temp = MAXVAL(slice)`` is a top-
    # level assign the existing reduce-emit dispatch handles, and the
    # outer expression sees a clean scalar load.
    "hlfir-lift-reduction-operands,"
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

# Sympy module-level attributes that turn user-source identifiers into
# parser hazards.  ``test`` / ``doctest`` are ``LazyFunction`` wrappers
# that fail sympify with ``cannot sympify object of type LazyFunction``
# whenever a string referencing them is parsed (interstate-edge
# expressions, memlet subsets, etc.).  The bridge renames any matching
# Fortran identifier to ``program_<name>`` at the SDFG layer; the binding
# emitter restores the original name on the Python wrapper.
_RESERVED_DACE_NAMES = frozenset({"test", "doctest"})

_DACE_NAME_PREFIX = "program_"


def _rename_reserved_collisions(sdfg) -> dict:
    """Walk ``sdfg.arrays`` / ``sdfg.symbols`` for entries whose name
    collides with a reserved sympy attribute and apply a deterministic
    ``program_<name>`` rename via ``sdfg.replace`` (which sweeps every
    memlet, code string, interstate-edge expression, and access node
    in lockstep).  Returns ``{user_fortran_name: dace_name}`` for the
    binding emitter; empty dict when nothing collided.
    """
    renames = {}
    for name in list(sdfg.arrays.keys()) + list(sdfg.symbols.keys()):
        if name in _RESERVED_DACE_NAMES:
            renames[name] = _DACE_NAME_PREFIX + name
    for old, new in renames.items():
        sdfg.replace(old, new)
    return renames


class SDFGBuilder:
    """Walks the HLFIR ASTNode tree and emits a DaCe SDFG.

    Public surface:
        builder = SDFGBuilder("code.hlfir")
        sdfg    = builder.build()

    After construction:
        self.variables   --  full VarInfo list from the bridge.
        self.arrays      --  {name: VarInfo} for rank>0 variables.
        self.symbols     --  {name: VarInfo} for scalars used in shapes / bounds /
                          control-flow conditions (pass 2b-2d of extract_vars).
        self.scalars     --  {name: VarInfo} for remaining scalars.
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
        separate ``.hlfir`` files  --  e.g. the ICON multi-module flow where
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
        # ``view_alias`` participates in the array dictionary so the
        # emitter routes accesses to it normally; ``add_descriptors``
        # registers it via ``sdfg.add_view`` (pointer alias of its
        # source array, no separate storage) and the ``acc`` factory
        # adds a per-state linking memlet so DaCe codegen knows
        # ``dd``'s reads/writes propagate to ``d``.
        self.arrays = {v.fortran_name: v for v in self.variables if v.role in ("array", "view_alias", "section_alias")}
        self.symbols = {v.fortran_name: v for v in self.variables if v.role == "symbol"}
        self.scalars = {v.fortran_name: v for v in self.variables if v.role == "scalar"}
        # Per-axis offset symbols: ``offset_<arr>_d<i>`` is the SDFG
        # symbol every memlet of array ``<arr>`` subtracts on dim ``i``.
        # Populated by ``add_descriptors`` from each VarInfo's
        # ``lower_bounds``.  Values are int (constant-folded by
        # ``sdfg.specialize``), str (substituted with another symbol
        # name), or ``None`` (unknown  --  symbol stays free, caller
        # passes it).
        self.offset_values: dict[str, int | str | None] = {}

    def build(self) -> SDFG:
        """Construct the SDFG, run the unconditional offset-symbol
        specialisation pass, and attach a frozen-signature snapshot.

        The snapshot is later verified by ``codegen.generate_code``
        before any C++ header gets emitted  --  any downstream
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
        # Stage source -> view-alias copy-in states ahead of the body
        # so reads on the alias see live source data.  After the body
        # we stage the reverse copy-out so writes propagate back.
        # The pre state becomes the SDFG's start block; the post state
        # is linked from the body's last state by edging through
        # ``ctx.cur``.
        ctx = _Ctx(sdfg, self)
        self._emit(ctx, self.ast, sdfg)
        ctx.flush(self)
        # User-source identifiers that collide with sympy module-level
        # names (``test`` / ``doctest`` are ``LazyFunction`` attributes
        # that crash sympify; bare letters like ``I`` resolve to
        # ``ImaginaryUnit``).  Rewrite to ``program_<name>`` so DaCe's
        # symbolic parsers stop reaching for the sympy attribute.  The
        # binding emitter consults ``self.dace_name_map`` to expose the
        # original Fortran name on the Python wrapper, so user-side
        # calls (``sdfg(test=arr)``) keep working.
        self.dace_name_map = _rename_reserved_collisions(sdfg)
        # Always-on post-emit substitution.  ``offset_values`` carries
        # two flavours of mapping: int constants (``offset_d_d0 = 50``
        # for ``dimension(50:54)``) and symbol aliases (``offset_d_d0 =
        # "arrsize"`` for ``dimension(arrsize:arrsize+4)``).  They take
        # different paths because ``sdfg.specialize`` only handles
        # constants  --  feeding it a string would land on ``add_constant``
        # and downstream casting tries ``int64("arrsize")`` and
        # ValueError-s.
        # TODO(future): replace the ``specialize`` call with
        # ``sdfg.replace_dict`` so the offset symbols get erased from
        # ``sdfg.symbols`` entirely (they currently linger as bound
        # constants and bloat the symbol table).  An attempt at this
        # broke ``test_fortran_frontend_type_array`` /
        # ``test_fortran_frontend_type_array2`` in ``type_test.py``:
        # for non-default lower bounds (``dimension(7:12)``) the
        # ``replace_dict`` substitution didn't apply uniformly to
        # every memlet subset, leaving raw-Fortran indices that went
        # out-of-bounds against the 0-based flat companion.  Needs a
        # careful audit of which property paths ``replace_dict``
        # walks (vs. what ``specialize`` does in-place via the
        # constants table) before re-trying.
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
        sdfg.validate()
        return sdfg

    def _register_constants(self, sdfg: SDFG) -> None:
        """Attach Flang's constant-pool data to the SDFG.

        Every ``VarInfo`` with non-empty ``const_data`` represents a
        ``_QQro.<shape>x<dtype>.<counter>`` global  --  the read-only
        backing for an array or scalar literal in the source.  The
        bridge has already added a transient descriptor for it via
        ``add_descriptors``; this hook attaches the dense values so
        DaCe's codegen materialises them into the binary.

        The data widens to ``double`` on the bridge side for
        transport; we narrow to the descriptor's actual dtype here
        and reshape back to the rank-N companion shape.  Scalar
        constants (rank 0, single value) are uncommon  --  Fortran
        ``parameter`` scalars typically inline as ``arith.constant``
         --  but the path supports them with a trivial 1-element array.
        """
        import numpy as np
        from dace.data import Scalar as _Scalar
        for v in self.variables:
            if not v.const_data:
                continue
            if v.fortran_name not in sdfg.arrays:
                continue
            desc = sdfg.arrays[v.fortran_name]
            np_dtype = desc.dtype.as_numpy_dtype()
            arr = np.asarray(v.const_data, dtype=np.float64).astype(np_dtype)
            # Scalar (rank 0) globals  --  e.g. ``real :: bob = 1`` at
            # module scope  --  pass through as a Python scalar so DaCe's
            # ``framecode.generate_constants`` writes a
            # ``constexpr <T> name = <val>`` (not the array form which
            # tries to ``sym2cpp`` a numpy array and chokes with
            # ``unhashable type: 'numpy.ndarray'``).
            if isinstance(desc, _Scalar) or v.rank == 0:
                if arr.size != 1:
                    continue
                sdfg.add_constant(v.fortran_name, arr.item(), desc)
                continue
            # Array globals: reshape from row-major doubles transport
            # to the descriptor's declared shape.
            shape = tuple(int(d) for d in desc.shape)
            if arr.size == int(np.prod(shape)):
                arr = arr.reshape(shape, order='C')
            sdfg.add_constant(v.fortran_name, arr, desc)

    def _run_post_gen_passes(self, sdfg: SDFG) -> None:
        """Run the post-generation cleanup passes that take a freshly-
        emitted bridge SDFG to its canonical shape.  See Stage 4b in
        ``dace/frontend/hlfir/README.md`` for the pipeline.

        Currently:
            * ``UniqueLoopIterators`` -- rewrites every ``LoopRegion``'s
              loop variable to a globally-unique ``_loop_it_<N>`` symbol
              and propagates the rename through the body.  Enabled with
              ``assign_loop_iterator_post_value=True``: bridge-emitted
              SDFGs land in Fortran callers that read the iterator after
              the loop end (gfortran/ifort/flang convention: one stride
              past the last attained value), so the pass also stages a
              postfix-assignment state for that read.
            * ``replace_length_one_arrays_with_scalars`` -- folds
              every length-1 ``Array`` on the SDFG signature down to a
              true ``Scalar``.  The bridge already emits scalar inputs
              directly as ``Scalar``; this pass cleans up leftover
              length-1 OUTPUTS and any local 1-element transients so
              callers can bind plain ``int`` / ``float`` instead of
              wrapping in a numpy 1-array.
        """
        from dace.sdfg.construction_utils import replace_length_one_arrays_with_scalars
        from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators

        # Empty-region cleanup: any ControlFlowRegion (LoopRegion,
        # ConditionalBlock branch, the top-level SDFG, etc.) that
        # ended up with zero internal blocks gets a single empty
        # state added.  Validation requires every CFG region to
        # have a defined start block; an empty region triggers
        # "Ambiguous starting block".  Such empties arise legitimately
        # from Fortran source  --  a ``do i = 1, N; <only-stripped-by-
        # flatten>; end do`` whose body became a no-op after
        # AoS+allocatable flattening, an empty IF branch, etc.
        # The empty state is semantically equivalent to the source
        # construct (a loop iterating over a no-op body is still
        # a no-op overall).
        for region in list(sdfg.all_control_flow_regions()):
            if len(list(region.nodes())) == 0:
                region.add_state("empty_body", is_start_block=True)

        uniq_loop_iter_pass = UniqueLoopIterators()
        uniq_loop_iter_pass.assign_loop_iterator_post_value = True
        uniq_loop_iter_pass.apply_pass(sdfg, {})
        # ``transient_only=True``: only fold LOCAL 1-element transients
        # (e.g. accumulators left as length-1 arrays by the bridge).  The
        # signature convention is preserved: ``intent(out)`` / ``inout``
        # scalars stay as length-1 ``Array`` so callers can pass a numpy
        # 1-element buffer to receive the value.  ``intent(in)`` /
        # ``VALUE`` scalars are already emitted as ``Scalar`` directly by
        # ``descriptors.py`` and don't need this pass.
        replace_length_one_arrays_with_scalars(sdfg, recursive=True, transient_only=True)

        # Zero-initialise every transient Scalar AND transient Array in
        # the SDFG (and every nested SDFG).  Fortran semantics: a local
        # REAL/INTEGER scalar or array is undefined until written -- but
        # for code paths that read it under data-dependent IF branches
        # without a guaranteed prior write (common after a kernel is
        # carved down -- the producer of a still-read local lives in a
        # deleted section), gfortran and the bridge see DIFFERENT
        # garbage: the gfortran reference is built with
        # ``-finit-local-zero`` (locals, scalar and array, start at 0),
        # while the bridge's transients are ``new``-allocated and hold
        # heap garbage (~1e228), causing divergent IF outcomes and
        # cascading drift.  Forcing zero-init on every transient pins
        # the bit pattern across both paths -- matching
        # ``-finit-local-zero`` -- and eliminates this class of
        # divergence.  ``setzero`` only adds an initialiser, so a
        # transient that *is* written-before-read is unaffected.
        import dace
        from dace.data import Array, Scalar
        for nsdfg in [sdfg] + [n.sdfg for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]:
            for state in nsdfg.all_states():
                for node in state.nodes():
                    if not isinstance(node, dace.nodes.AccessNode):
                        continue
                    desc = nsdfg.arrays.get(node.data)
                    if isinstance(desc, (Scalar, Array)) and desc.transient:
                        node.setzero = True

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
        # Reverse the rename map so we can recover the user-source
        # Fortran name from the SDFG-internal name.  Empty dict when no
        # reserved-name collision fired, so the lookup becomes a no-op.
        dace_to_user = {v: k for k, v in getattr(self, 'dace_name_map', {}).items()}
        for sdfg_name_, desc in sdfg.arglist().items():
            user_key = dace_to_user.get(sdfg_name_, sdfg_name_)
            v = (self.arrays.get(user_key) or self.symbols.get(user_key) or self.scalars.get(user_key))
            if user_key in self.symbols:
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
                    fortran_name=v.fortran_name if v is not None else user_key,
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
        loop variable names (``jk_0``, ``jc_1``, ``jk_2``, ...) never
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
        """Recursive dispatcher  --  maps each ASTNode.kind to its emitter."""
        for n in nodes:
            fn = self._EMIT_DISPATCH.get(n.kind)
            if fn is not None:
                fn(self, ctx, n, region)
            # "call" currently has no emitter  --  nested SDFG / library node
            # is a future feature (Phase 4).

    # Scalar-assign is called from _Ctx.flush; keep it as a method on the
    # builder for that caller's convenience.
    def emit_scalar_assign(self, state, target: str, value: str):
        emit_scalar_assign(self, state, target, value)


def generate_sdfg(path: str = None, *, pipeline: str = None, entry: str = None, hlfir_files=None) -> SDFG:
    """Build an SDFG from one or several HLFIR files.

    Single-file form (back-compat):
        ``generate_sdfg("code.hlfir")``  --  parses + DEFAULT_PIPELINE.

    Multi-file form (ICON-style linked entry):
        ``generate_sdfg(entry="_QPkernel", hlfir_files=[...])``  --  parses
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
