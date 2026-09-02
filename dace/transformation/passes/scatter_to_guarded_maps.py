# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end pass: detect scatter loops, guard their index arrays, parallelize.

A *scatter loop* is a ``LoopRegion`` whose body writes to a non-transient array at
an index of the form ``arr[idx[f(i)]]`` -- the write slot is data-dependent
through an index array ``idx`` read at a (possibly strided) function of the loop
variable. ``LoopToMap`` refuses such loops by default because two iterations may
write the same slot; the user's contract is that ``idx`` is a permutation (no
duplicates -> no write-write race), and ``LoopToMap``'s ``permissive`` mode lifts
the loop under that assumption.

This pass operationalises that contract end-to-end:

1. **Detect** every scatter loop in the SDFG (see
   :func:`_scatter_idx_arrays_for_loop`), recognising the three lowered forms a
   ``out[idx[f(i)]] = ...`` scatter takes: an interstate-bound index
   (``sym := idx[f(i)]`` used in a write subset), an inline-subscript index
   (``idx[f(i)]`` written literally inside a write-memlet subset), and a
   nested-SDFG data-dependent write whose index traces back to an integer array
   read at ``[f(i)]``. ``f(i)`` is any expression referencing the loop variable,
   so both unit-stride (``idx[i]``) and symbolic-stride (``idx[SSYM*i]``)
   scatters are covered. The union of source-array names is the set of ``idx``
   arrays.

   A write whose slot takes MORE THAN ONE index (``out[xi[j], yi[j]]``) is not covered by any of
   the three: see :func:`joint_scatter_writes_for_loop`, which keys it across all of its
   dimensions instead, because per-array guards ask each index to be a permutation on its own and
   that is stronger than the loop needs.

   An indirect write carrying a WCR is skipped by all four (see
   :func:`indirect_write_needs_injectivity`): an accumulation is correct under colliding indices,
   so demanding a permutation of it would abort on inputs the program computes correctly.
2. **Guard** each detected ``idx`` array via
   :func:`~dace.transformation.passes.scatter_conflict_guard.insert_scatter_guard`,
   which inserts an ``IntegerSort`` + adjacent-equal-pair check + ``std::abort()``
   at the earliest legal CFG state.
3. **Parallelize** by applying ``LoopToMap`` in ``permissive`` mode, which lifts
   the scatter loops (and any other previously refused permissive cases) into
   parallel Maps.

The ordering is intentional: guards are emitted *before* permissive lifts, so on
collision the abort fires before any consumer reads the corrupted output.
"""
import ast
import copy
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import dace
from dace import SDFG, SDFGState, data, dtypes, memlet as mm, properties, subsets, symbolic
from dace.frontend.python import astutils
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.scatter_conflict_guard import (ScatterIndexSlice, build_guard_states,
                                                               insert_scatter_guard)
from dace.transformation.passes.vectorization.utils.map_predicates import NO_VECTORIZE_MARKER


@properties.make_properties
@xf.explicit_cf_compatible
class ScatterToGuardedMaps(ppl.Pass):
    """Detect scatter loops, insert per-array runtime guards, then permissively lift to maps.

    Two collision policies are supported via :attr:`emit_unparallelized_else_branch`:

    - ``False`` (default): the guard's check tasklet calls ``std::abort()``
      whenever a duplicate is detected; the parallelised Map runs unconditionally
      afterwards. The contract is "permutation or abort" -- callers committed to
      that contract get the simpler CFG.
    - ``True``: the guard emits only the sort + duplicate-count steps (no trap)
      and the scatter loop is wrapped in a ``ConditionalBlock`` keyed on the
      duplicate-count symbol. The ``True`` branch keeps a deep copy of the
      original sequential ``LoopRegion``; the ``False`` branch holds the
      ``LoopToMap``-lifted parallel Map. The check tasklet routes at runtime,
      so collisions degrade to sequential execution rather than aborting.

    Idempotence: the underlying guard utility refuses to emit a second guard for the
    same ``idx`` array (the ``_scatter_guard_sorted_<name>`` transient acts as the
    presence marker). Re-running this pass on an SDFG it has already guarded is a
    no-op for the guard step; the ``LoopToMap`` step still re-applies and is itself
    idempotent on already-lifted Maps.
    """

    CATEGORY: str = 'Optimization Preparation'

    emit_unparallelized_else_branch = properties.Property(
        dtype=bool,
        default=False,
        desc="When True, emit a ``ConditionalBlock`` dispatching at runtime on "
        "the duplicate-count symbol: the True branch runs a sequential clone "
        "of the original scatter loop; the False branch runs the parallel "
        "Map lift. The duplicate-trap is suppressed; collisions degrade to "
        "sequential execution instead of ``std::abort()``.",
    )

    assume_no_conflicts = properties.Property(
        dtype=bool,
        default=False,
        desc="When True, ASSUME every scatter ``idx`` array is a permutation "
        "(no duplicate targets): skip the sort + duplicate-count guard entirely "
        "and lift the scatter loop to an unconditional parallel Map. Unsound if "
        "the assumption is violated at runtime (write races); the caller owns "
        "that contract. Takes precedence over ``emit_unparallelized_else_branch``.",
    )

    def __init__(self, emit_unparallelized_else_branch: bool = False, assume_no_conflicts: bool = False):
        super().__init__()
        self.emit_unparallelized_else_branch = emit_unparallelized_else_branch
        self.assume_no_conflicts = assume_no_conflicts

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        """Run the full pipeline. Returns the number of distinct ``idx`` arrays guarded,
        or ``None`` if no scatter loop was found.
        """
        from dace.transformation.interstate.loop_to_map import LoopToMap

        scatter_loops, idx_arrays, sliced_guards, joint_writes = detect_scatter_loops_and_idx_arrays(
            sdfg, allow_hoisted_joint=not self.emit_unparallelized_else_branch)

        if self.assume_no_conflicts:
            # Caller asserts every idx array is a permutation: skip the sort +
            # duplicate-count guard entirely and lift each scatter loop to an
            # unconditional parallel Map (no ConditionalBlock, no trap).
            for loop in scatter_loops:
                parent = loop.parent_graph
                if parent is None or loop not in parent.nodes():
                    continue
                instance = LoopToMap()
                instance.loop = loop
                try:
                    instance.apply(parent, _owning_sdfg(sdfg, loop))
                except Exception:
                    pass
            return (len(idx_arrays) + len(sliced_guards) + len(joint_writes)) or None

        # A multi-dimensional indirect write is keyed across ALL of its dimensions and guarded on
        # that key, right before its own loop -- the key is recomputed whenever the loop is
        # re-entered, so it cannot be hoisted to the key array's definition the way a written-once
        # index array can. Emitted first so the count symbols exist for the dispatcher below.
        joint_dup_syms: dict = {}
        for loop, write in joint_writes:
            # The key array and the guard states belong to the sdfg named by the write's plan --
            # the loop's own owner normally, and the region above it when that owner is a trivial
            # map wrapper. Putting them anywhere else leaves the fill state naming a descriptor its
            # own sdfg does not have ("Data descriptor _scatter_joint_key_... not defined in SDFG"
            # -- npbench mandelbrot2 under the canonicalize+vectorize pipeline).
            trap_sym = guard_joint_scatter_write(sdfg, loop, write, emit_trap=not self.emit_unparallelized_else_branch)
            if trap_sym is not None:
                joint_dup_syms.setdefault(id(loop), []).append(trap_sym)

        # Track each idx_array's duplicate-count symbol so the else-branch
        # dispatcher knows which symbol to gate on per scatter loop. None when
        # the trap mode is on.
        dup_count_syms: dict = {}
        for idx_name in sorted(idx_arrays):
            try:
                trap_sym = insert_scatter_guard(sdfg, idx_name, emit_trap=not self.emit_unparallelized_else_branch)
                if trap_sym is not None:
                    dup_count_syms[idx_name] = trap_sym
            except ValueError as exc:
                if 'already exists' not in str(exc):
                    raise

        # A rank>=2 idx array classified to a single contiguous varying dimension: guarded at
        # the loop's OWNING sdfg (may be nested -- the slice's fixed-dim expressions can
        # reference symbols, e.g. an outer loop variable, that only exist there), keyed by
        # (owner sdfg, name) since the same array name can be sliced differently per scope.
        # The guard STATES themselves are hoisted to the outermost enclosing loop that
        # natively defines those symbols (see :func:`_hoist_slice_region`): splicing them into
        # the loop's own immediate owning sdfg would turn that sdfg's trivial single-state
        # LoopToMap wrapper multi-state, blocking the later inline+collapse of the surrounding
        # map nest.
        sliced_dup_syms: dict = {}
        for loop, idx_name, index_slice in sliced_guards:
            owner_sdfg = _owning_sdfg(sdfg, loop)
            host_region = _hoist_slice_region(sdfg, owner_sdfg, index_slice)
            host_sdfg = host_region.sdfg
            if idx_name not in host_sdfg.arrays:
                host_region, host_sdfg = owner_sdfg, owner_sdfg
            try:
                trap_sym = insert_scatter_guard(host_sdfg,
                                                idx_name,
                                                emit_trap=not self.emit_unparallelized_else_branch,
                                                index_slice=index_slice,
                                                region=host_region)
                if trap_sym is not None:
                    sliced_dup_syms[(id(loop), idx_name)] = trap_sym
            except ValueError as exc:
                if 'already exists' not in str(exc):
                    raise

        for loop in scatter_loops:
            parent = loop.parent_graph
            if parent is None or loop not in parent.nodes():
                continue  # already removed by a sibling lift
            owner_sdfg = _owning_sdfg(sdfg, loop)

            if self.emit_unparallelized_else_branch and joint_dup_syms.get(id(loop)):
                # A joint key already answers for every dimension of this loop's writes, so its
                # counts alone decide the branch -- the per-array symbols below are empty here.
                cond = ' + '.join(joint_dup_syms[id(loop)]) + ' > 0'
                _wrap_loop_in_dispatcher(parent, loop, cond, LoopToMap)
                continue

            if self.emit_unparallelized_else_branch and (dup_count_syms or sliced_dup_syms):
                # Find the dup-count symbol for any idx array this loop
                # scatters into. Loops with multiple idx arrays would need ALL
                # of them to be conflict-free for the parallel branch to be
                # safe; we OR the counts together so any positive count routes
                # to the sequential branch.
                loop_idx = _scatter_idx_arrays_for_loop(loop, owner_sdfg)
                loop_idx_syms = [dup_count_syms[i] for i in loop_idx if i in dup_count_syms]
                loop_idx_syms += [sliced_dup_syms[(id(loop), i)] for i in loop_idx if (id(loop), i) in sliced_dup_syms]
                if loop_idx_syms:
                    cond = ' + '.join(loop_idx_syms) + ' > 0'
                    _wrap_loop_in_dispatcher(parent, loop, cond, LoopToMap)
                    continue

            instance = LoopToMap()
            instance.loop = loop
            try:
                instance.apply(parent, owner_sdfg)
            except Exception:
                pass
        return (len(idx_arrays) + len(sliced_guards) + len(joint_writes)) or None


def detect_scatter_idx_arrays(sdfg: SDFG) -> Set[str]:
    """Find every ``idx`` array name used as an indirect-write index in any LoopRegion.

    See :func:`detect_scatter_loops_and_idx_arrays` for the underlying scan; this
    helper drops the loops set and returns only the idx-array names.
    """
    _, idx_arrays, sliced_guards, _joint = detect_scatter_loops_and_idx_arrays(sdfg)
    return idx_arrays | {idx_name for _loop, idx_name, _index_slice in sliced_guards}


def detect_scatter_loops_and_idx_arrays(sdfg: SDFG, allow_hoisted_joint: bool = True):
    """Scan ``sdfg`` (and nested SDFGs) for scatter loops; return
    ``(scatter_loops, idx_arrays, sliced_guards, joint_writes)``.

    A ``LoopRegion`` qualifies as a scatter loop iff any interstate edge in the
    region binds a symbol via ``sym := arr[loop_var]`` AND a write-memlet to a
    non-transient array inside the region's body references that symbol.

    A write whose target slot takes more than one index is reported separately, in
    ``joint_writes``, and its index arrays are deliberately kept OUT of ``idx_arrays``: guarding
    them one at a time asks each to be a permutation on its own, which is stronger than the loop
    needs and false for the shape this exists to handle (see
    :func:`joint_scatter_writes_for_loop`). A loop whose multi-dimensional indirect write cannot be
    keyed is dropped from the scan entirely rather than falling back to that stronger check.

    :param sdfg: The SDFG to scan; nested SDFGs are walked too.
    :param allow_hoisted_joint: whether a joint guard may be HOISTED out of a trivial map wrapper
        (:func:`joint_guard_plan`). False under the dispatcher policy, whose duplicate-count symbol
        is read inside the loop's own sdfg and would not be defined once the guard moves out.
    :returns: ``(list[LoopRegion], set[str], list[tuple], list[tuple])`` -- deterministic-order list
              of the scatter ``LoopRegion`` instances; the set of 1-D ``idx`` array names (guarded
              once, at ``sdfg``, over the whole array); the deterministic-order list of
              ``(loop, idx_name, ScatterIndexSlice)`` triples for a rank>=2 ``idx`` array whose
              subscript classified to a single contiguous varying dimension (see
              :func:`_classify_index_slice`) -- each guarded at its own loop's owning sdfg; and the
              ``(loop, JointScatterWrite)`` pairs to key across all of their dimensions.
    """
    scatter_loops: list = []
    idx_arrays: Set[str] = set()
    sliced_guards: list = []
    joint_writes: list = []
    for sd in sdfg.all_sdfgs_recursive():
        for region in sd.all_control_flow_regions():
            if not (isinstance(region, LoopRegion) and region.loop_variable):
                continue
            joint = joint_scatter_writes_for_loop(region, sd)
            if joint is None:
                continue  # an indirect write this cannot key -- leave the loop sequential
            if joint:
                plans = [joint_guard_plan(sdfg, region, w) for w in joint]
                if all(p is not None and (allow_hoisted_joint or not p.map_dims) for p in plans):
                    scatter_loops.append(region)
                    joint_writes += [(region, w) for w in joint]
                    continue
            # A keyable write whose guard cannot be placed (see :func:`joint_guard_plan`) falls
            # through to the per-index scan below, and is left sequential when that finds nothing
            # either. Those guards are strictly stronger -- each index array must be a permutation
            # on its own -- so this can trap a program whose index PAIRS are distinct, which is the
            # case the joint key exists for. It buys back the enclosing nest's rank.
            loop_arrays = _scatter_idx_targets_for_loop(region, sd)
            if not loop_arrays:
                continue
            scatter_loops.append(region)
            for idx_name, (_tgts, index_slice) in loop_arrays.items():
                if index_slice is None:
                    idx_arrays.add(idx_name)
                else:
                    sliced_guards.append((region, idx_name, index_slice))
    sliced_guards.sort(key=lambda t: (t[0].label, t[1]))
    return scatter_loops, idx_arrays, sliced_guards, joint_writes


def scatter_target_arrays(sdfg: SDFG, idx_name: str) -> Set[str]:
    """Names of the arrays ``idx_name`` scatters into, across ``sdfg``'s own scatter loops.

    The scatter guard sizes its value-indexed tag array by these arrays' domains (see
    :func:`~dace.transformation.passes.scatter_conflict_guard.scatter_index_domain`), so the
    answer must come from the same detector that decides ``idx_name`` needs guarding.

    Top-level only: a nested SDFG names the same array through its own connector, which does not
    resolve against ``sdfg.arrays``, and the guard itself is emitted at the top level.

    :param sdfg: The SDFG to scan.
    :param idx_name: The scatter index array.
    :returns: The set of scattered-into array names (empty if ``idx_name`` drives no scatter).
    """
    targets: Set[str] = set()
    for region in sdfg.all_control_flow_regions():
        if not (isinstance(region, LoopRegion) and region.loop_variable):
            continue
        entry = _scatter_idx_targets_for_loop(region, sdfg).get(idx_name)
        if entry is not None:
            targets |= entry[0]
    return targets


def indirect_write_needs_injectivity(write: mm.Memlet) -> bool:
    """Whether an indirect write has to be proven collision-free before its loop may be lifted.

    A plain write does: two iterations landing on the same slot race, and which one survives is
    undefined. A WCR write does NOT. It is a read-modify-write the codegen lowers to an atomic
    combine, so colliding iterations fold into the slot one after another and the reducer's
    associativity makes the order immaterial -- ``bins[ip[i]] (+)= src[i]`` is correct for any
    ``ip`` whatever. Guarding one would demand a permutation the program never needed and abort
    on inputs it computes correctly.

    :param write: The write memlet.
    :returns: True for a plain write, False for an accumulation.
    """
    return write.wcr is None


def nested_write_is_accumulation(nsdfg_node: nodes.NestedSDFG, out_conn: str) -> bool:
    """Whether every write to ``out_conn`` INSIDE ``nsdfg_node`` carries a WCR.

    A ``dace.map`` scatter ``hist[bin[i]] += w[i]`` lowers to a NestedSDFG holding the WCR write
    against a write-only output connector while the connector's own outer edge stays plain, so
    the outer memlet alone does not say whether the write accumulates.

    :param nsdfg_node: The nested SDFG performing the write.
    :param out_conn: The output connector carrying it.
    :returns: True iff ``out_conn`` is written at least once and every such write is a WCR.
    """
    writes = [
        e.data for st in nsdfg_node.sdfg.all_states() for dn in st.data_nodes() if dn.data == out_conn
        for e in st.in_edges(dn) if e.data is not None and not e.data.is_empty()
    ]
    return bool(writes) and all(not indirect_write_needs_injectivity(m) for m in writes)


def _scatter_idx_arrays_for_loop(region: LoopRegion, sdfg: SDFG) -> Set[str]:
    """Return the scatter index-array names driving an indirect WRITE in ``region``.

    Names only; see :func:`_scatter_idx_targets_for_loop` for the scan and for which arrays
    each index writes through.
    """
    return set(_scatter_idx_targets_for_loop(region, sdfg))


def _scatter_idx_targets_for_loop(region: LoopRegion,
                                  sdfg: SDFG) -> Dict[str, Tuple[Set[str], Optional[ScatterIndexSlice]]]:
    """Map each scatter index-array name driving an indirect WRITE in ``region`` to the arrays
    it writes through, plus (for a rank>=2 index array) the 1-D window the guard should scan.

    Recognises the three lowered forms an ``out[idx[f(i)]] = ...`` scatter takes,
    where ``f(i)`` is any expression referencing the loop variable (a bare
    ``idx[i]`` or a strided ``idx[c*i + d]``):

    1. **Interstate-bound index** -- ``sym := idx[f(i)]`` on a loop interstate
       edge, with ``sym`` referenced in a write-memlet subset to a non-transient
       array (TSVC ``s4113``/``s491``/``vas`` and the symbolic-stride
       ``s4113_ssym``).
    2. **Inline-subscript index** -- the index array appears literally inside a
       write-memlet subset, ``out[idx[f(i)]]`` (the symbolic-stride ``vas_ssym``).
    3. **Nested-SDFG data-dependent write** -- a ``NestedSDFG`` writes a
       non-transient array with a data-dependent (fewer accesses than the subset
       spans) memlet whose write index traces back to an integer array read at
       ``[f(i)]`` (``ext_scatter_store``, lowered from a ``dace.map`` scatter).

    A 1-D index array is always included (``index slice = None`` -- the guard scans the whole
    declared array). A rank>=2 index array (forms 1/2 only -- form 3 has no subscript AST to
    classify) is included only when :func:`_classify_index_slice` pins its subscript to a
    single contiguous varying dimension; otherwise it is dropped and the loop stays un-lifted
    for that array.

    :param region: The candidate loop region.
    :param sdfg: The SDFG owning ``region``'s arrays.
    :returns: ``{idx array name: (arrays scattered into, index slice or None)}`` (empty if
              ``region`` is not a scatter).
    """
    loop_var = region.loop_variable
    bindings = _collect_indirect_bindings(region, sdfg)
    loop_arrays: Dict[str, Set[str]] = {}
    dim_nodes_by_arr: Dict[str, List[ast.AST]] = {}
    for state in region.all_states():
        for node in state.data_nodes():
            if state.in_degree(node) == 0:
                continue
            desc = sdfg.arrays.get(node.data)
            if desc is None or desc.transient:
                continue
            for e in state.in_edges(node):
                if e.data is None or e.data.subset is None or not indirect_write_needs_injectivity(e.data):
                    continue
                # Form 1: interstate-bound index symbol referenced in the write subset.
                for sym in e.data.subset.free_symbols:
                    binding = bindings.get(str(sym))
                    if binding is not None:
                        arr, dim_nodes = binding
                        loop_arrays.setdefault(arr, set()).add(node.data)
                        dim_nodes_by_arr.setdefault(arr, dim_nodes)
                # Form 2: index array inline-subscripted inside the write subset.
                for arr, dim_nodes in _inline_indirect_idx_arrays(e.data.subset, loop_var, sdfg).items():
                    loop_arrays.setdefault(arr, set()).add(node.data)
                    dim_nodes_by_arr.setdefault(arr, dim_nodes)
        # Form 3: nested-SDFG data-dependent write.
        for arr, tgts in _nested_dynamic_scatter_idx_arrays(state, sdfg, loop_var).items():
            loop_arrays.setdefault(arr, set()).update(tgts)

    result: Dict[str, Tuple[Set[str], Optional[ScatterIndexSlice]]] = {}
    for arr, tgts in loop_arrays.items():
        desc = sdfg.arrays[arr]
        if len(desc.shape) == 1:
            result[arr] = (tgts, None)
            continue
        dim_nodes = dim_nodes_by_arr.get(arr)
        if dim_nodes is None:
            continue  # form 3 only -- no subscript AST to classify, stays excluded.
        index_slice = _classify_index_slice(desc, dim_nodes, region)
        if index_slice is not None:
            result[arr] = (tgts, index_slice)
    return result


def _classify_index_slice(desc: data.Array, dim_nodes: List[ast.AST],
                          region: LoopRegion) -> Optional[ScatterIndexSlice]:
    """Classify a rank>=2 index-array subscript ``arr[dim_nodes...]`` against ``region``'s loop
    variable: accepted iff exactly one dimension is affine in the loop variable and every other
    dimension is loop-invariant. ``ScatterConflictCheck`` scans its input through a flat
    pointer, so the varying dimension's per-iteration element stride (its affine coefficient,
    times the loop's own stride, times ``desc.strides[dim]``) must additionally resolve to 1 --
    a genuinely contiguous window. Returns ``None`` (leave the loop un-lifted) on any mismatch.
    """
    if len(dim_nodes) != len(desc.shape):
        return None
    loop_var = region.loop_variable
    varying = [
        d for d, node in enumerate(dim_nodes) if loop_var in {n.id
                                                              for n in ast.walk(node) if isinstance(n, ast.Name)}
    ]
    if len(varying) != 1:
        return None
    dim = varying[0]

    init = loop_analysis.get_init_assignment(region)
    end = loop_analysis.get_loop_end(region)
    lstride = loop_analysis.get_loop_stride(region)
    if init is None or end is None or lstride is None:
        return None

    j = symbolic.pystr_to_symbolic(loop_var)
    dim_expr = symbolic.pystr_to_symbolic(astutils.unparse(dim_nodes[dim]))
    coeff = dim_expr.coeff(j, 1)
    const = dim_expr.coeff(j, 0)
    if symbolic.simplify(dim_expr - (coeff * j + const)) != 0:
        return None  # not affine in the loop variable

    elem_stride = symbolic.simplify(coeff * lstride * desc.strides[dim])
    if symbolic.simplify(elem_stride - 1) != 0:
        return None  # not contiguous -- unsafe for the flat-pointer conflict-check scan

    extent = symbolic.simplify(symbolic.int_floor(end - init, lstride) + 1)
    offset = symbolic.simplify(coeff * init + const)
    fixed: Dict[int, str] = {d: astutils.unparse(node) for d, node in enumerate(dim_nodes) if d != dim}
    return ScatterIndexSlice(dim=dim, offset=str(offset), extent=str(extent), stride='1', fixed=fixed)


class JointScatterWrite(NamedTuple):
    """One indirect write whose target SLOT is chosen by more than one subscript dimension.

    ``dim_exprs`` holds the target's index per dimension, written in terms of the loop variable and
    of index-array reads (``Xi_0[j]``); ``mask_expr`` is the read deciding whether the write happens
    at all, ``None`` for an unconditional one.
    """
    target: str
    dim_exprs: Tuple[str, ...]
    mask_expr: Optional[str]


def joint_key_read_arrays(write: JointScatterWrite) -> Set[str]:
    """Every array name subscripted inside ``write``'s index and mask expressions."""
    names: Set[str] = set()
    for expr in write.dim_exprs + ((write.mask_expr, ) if write.mask_expr else ()):
        for node in ast.walk(ast.parse(expr, mode='eval')):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                names.add(node.value.id)
    return names


class JointGuardPlan(NamedTuple):
    """Where a joint guard's states go, and which enclosing parallel scopes its key must cover.

    ``map_dims`` is one ``(param, extent)`` per map scope crossed on the way out. Crossing one
    means the guard now answers for every value of that param at once, so the key grows a
    dimension for it -- flattened into the single dimension the conflict check reads. The order is
    outermost-first within one climb step and inner-step-first across steps; either way each param
    is paired with its own radix, so the flat position stays a bijection, which is all a key used
    only for equality needs.
    """
    host_sdfg: SDFG
    host_region: ControlFlowRegion
    anchor: ControlFlowBlock
    map_dims: Tuple[Tuple[str, symbolic.SymbolicType], ...]


def owner_is_trivial_map_wrapper(owner_sdfg: SDFG) -> bool:
    """Whether ``owner_sdfg`` is a ONE-BLOCK nested sdfg sitting inside a map scope.

    That is the shape ``LoopToMap`` leaves behind, and states may not be added to it: a
    multi-state nested sdfg in a map scope cannot be inlined, so the surrounding map nest never
    collapses and every map in it stays one-dimensional (``MarkTileDims`` then refuses the whole
    nest for want of ``K`` params). The guard has to be placed further out -- see
    :func:`joint_guard_plan`.
    """
    nsdfg = owner_sdfg.parent_nsdfg_node
    if nsdfg is None or owner_sdfg.parent is None:
        return False
    if owner_sdfg.parent.entry_node(nsdfg) is None:
        return False
    return len(owner_sdfg.nodes()) == 1


def map_scope_chain(state: SDFGState, node: nodes.Node) -> Optional[List[nodes.MapEntry]]:
    """The map entries enclosing ``node`` in ``state``, outermost first.

    :returns: the chain, or ``None`` when a scope on it is not a map -- a Consume scope has no
        ``map`` to read an extent off, and its iteration count is a runtime stream property rather
        than a range, so a key cannot be sized to cover it.
    """
    chain: List[nodes.MapEntry] = []
    entry = state.entry_node(node)
    while entry is not None:
        if not isinstance(entry, nodes.MapEntry):
            return None
        chain.append(entry)
        entry = state.entry_node(entry)
    chain.reverse()
    return chain


def joint_guard_plan(root: SDFG, loop: LoopRegion, write: JointScatterWrite) -> Optional[JointGuardPlan]:
    """Where to put the guard for ``loop``, climbing out of any trivial map wrapper on the way.

    Placing the states beside the loop is the normal answer and needs no climb. When the loop's
    owning sdfg is one of the wrappers :func:`owner_is_trivial_map_wrapper` describes, the states
    move to the region holding that wrapper's own state, and the key grows one dimension per map
    scope crossed -- the guard is now answering for the whole parallel scope at once, which is the
    right question anyway: the crossed maps are already parallel, so an injective key over the
    joint space is exactly what makes the write race-free.

    Climbing is refused, and the caller left to fall back to the per-index guards, when the move
    would change what the key MEANS rather than merely where it is computed:

    * a crossed map that is not ``0:N:1`` -- the flattened key position would need the general
      affine inverse, and no shape here has ever needed it;
    * a nested sdfg that RENAMES a symbol the key reads (its ``symbol_mapping`` is not the
      identity there), since the index expressions are written in the inner names;
    * a target or index array that the host sdfg does not have under the same name, or that is a
      View there -- the read would bind to nothing.

    :returns: the plan, or ``None`` when the guard cannot be placed without changing its meaning.
    """
    owner = _owning_sdfg(root, loop)
    plan = JointGuardPlan(owner, loop.parent_graph, loop, ())
    if not owner_is_trivial_map_wrapper(owner):
        return plan

    free: Set[str] = set()
    for expr in write.dim_exprs + ((write.mask_expr, ) if write.mask_expr else ()):
        free |= {str(sym) for sym in symbolic.pystr_to_symbolic(expr).free_symbols}
    free.discard(loop.loop_variable)

    current = owner
    dims: List[Tuple[str, symbolic.SymbolicType]] = []
    while owner_is_trivial_map_wrapper(current):
        nsdfg = current.parent_nsdfg_node
        state = current.parent
        if any(str(nsdfg.symbol_mapping.get(sym, sym)) != sym for sym in sorted(free)):
            return None
        chain = map_scope_chain(state, nsdfg)
        if chain is None:
            return None
        for entry in chain:
            for param, (lb, ub, step) in zip(entry.map.params, entry.map.range.ranges):
                if symbolic.simplify(lb) != 0 or symbolic.simplify(step) != 1:
                    return None
                dims.append((param, symbolic.simplify(ub + 1)))
        current = state.sdfg
        plan = JointGuardPlan(current, state.parent_graph, state, tuple(dims))

    # Parsed last: every refusal above is cheaper than walking the index expressions' ASTs.
    for name in sorted({write.target} | joint_key_read_arrays(write)):
        desc = plan.host_sdfg.arrays.get(name)
        if desc is None or isinstance(desc, data.View):
            return None
    return plan


def point_index_expressions(subset) -> Optional[List[str]]:
    """Per-dimension index of a subset addressing exactly one element, or ``None`` if it spans."""
    exprs: List[str] = []
    for rb, re_, _ in subset.ndrange():
        if symbolic.simplify(symbolic.pystr_to_symbolic(str(rb)) - symbolic.pystr_to_symbolic(str(re_))) != 0:
            return None
        exprs.append(str(rb))
    return exprs


def substitute_indirect_bindings(expr: str, bindings: Dict[str, Tuple[str, List[ast.AST]]]) -> str:
    """Rewrite every bound scatter symbol in ``expr`` back into the index-array read it stands for.

    The frontend hoists a scatter index onto an interstate edge (``__sym := Xi_0[j]``) and leaves
    only the symbol in the write subset, so a subset read straight off the edge names symbols rather
    than the data they came from.
    """

    class Expand(ast.NodeTransformer):

        def visit_Name(self, node: ast.Name):
            binding = bindings.get(node.id)
            if binding is None:
                return node
            arr, dim_nodes = binding
            return ast.parse(f"{arr}[{', '.join(astutils.unparse(d) for d in dim_nodes)}]", mode='eval').body

    return astutils.unparse(Expand().visit(ast.parse(expr, mode='eval').body))


def indirect_reads(expr: str, sdfg: SDFG) -> List[Tuple[str, str]]:
    """``(array, index expression)`` for every data-array subscript appearing in ``expr``."""
    reads: List[Tuple[str, str]] = []
    for node in ast.walk(ast.parse(expr, mode='eval').body):
        if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)):
            continue
        if node.value.id not in sdfg.arrays:
            continue
        idx = node.slice
        if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
            idx = idx.value
        reads.append((node.value.id, astutils.unparse(idx)))
    return reads


def region_assigned_symbols(region: LoopRegion) -> Set[str]:
    """Symbols an interstate edge inside ``region`` assigns, i.e. those that vary per iteration."""
    return {lhs for e in region.all_interstate_edges() for lhs in (e.data.assignments or {})}


def key_map_can_read(expr: str, sdfg: SDFG, loop_var: str, varying: Set[str]) -> bool:
    """Whether ``expr`` is computable outside the loop body, one value per iteration.

    Array subscripts and the loop variable are readable from the key map; anything else must hold
    still for the loop's whole execution, since the key map has no access to a value that a body
    interstate edge recomputes on the way to the write.
    """
    tree = ast.parse(expr, mode='eval').body
    subscripted = {
        node.value.id
        for node in ast.walk(tree) if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id in subscripted or node.id == loop_var:
            continue
        if node.id in varying:
            return False
    return True


def enclosing_branch_condition(block, region: LoopRegion,
                               bindings: Dict[str, Tuple[str, List[ast.AST]]]) -> Tuple[Optional[str], bool]:
    """The condition gating ``block`` inside ``region``, rewritten into index-array reads.

    A masked-off iteration writes nothing, so it must not be keyed onto a slot another iteration
    writes -- which means the mask has to be READABLE from the key map, not merely known to exist.
    Returns ``(expr, True)`` when every enclosing conditional resolves, ``(None, False)`` when one
    does not: an ``else`` branch, whose condition is the negation of a list this does not
    reconstruct.
    """
    conds: List[str] = []
    cur = block
    while cur is not None and cur is not region:
        parent = cur.parent_graph
        if isinstance(parent, ConditionalBlock):
            branch = next((cond for cond, body in parent.branches if body is cur), None)
            if branch is None:
                return None, False
            conds.append(substitute_indirect_bindings(branch.as_string, bindings))
        cur = parent
    if not conds:
        return None, True
    return ' and '.join(f'({c})' for c in conds), True


def joint_scatter_writes_for_loop(region: LoopRegion, sdfg: SDFG) -> Optional[List[JointScatterWrite]]:
    """Indirect writes in ``region`` whose target slot is picked by more than one dimension.

    ``N_[Xiv[j], Yiv[j]] = ...`` is the shape. Guarding ``Xiv`` and ``Yiv`` separately -- what the
    one-array-per-guard path does -- demands each be a permutation on its own, far more than the
    loop needs: mandelbrot2's ``Xiv`` holds ``XN`` distinct values across ``XN*YN`` entries while
    the PAIR is injective. Only the flattened slot an iteration writes says whether two iterations
    collide, so a write with any indirect dimension is keyed across all of its dimensions.

    :returns: The writes to key jointly (possibly empty), or ``None`` when the loop has an indirect
              multi-dimensional write that cannot be keyed -- an unresolved mask, a subset that
              spans, or an index built from a symbol the body recomputes. The caller must then leave
              the loop alone: falling back to per-array guards would trap a correct program.
    """
    loop_var = region.loop_variable
    bindings = _collect_indirect_bindings(region, sdfg)
    varying = region_assigned_symbols(region) - set(bindings)
    writes: List[JointScatterWrite] = []
    for state in region.all_states():
        for node in state.data_nodes():
            desc = sdfg.arrays.get(node.data)
            if not isinstance(desc, data.Array) or len(desc.shape) < 2:
                continue
            for e in state.in_edges(node):
                if e.data is None or e.data.is_empty() or not indirect_write_needs_injectivity(e.data):
                    continue
                subset = e.data.dst_subset if e.data.dst_subset is not None else e.data.subset
                if subset is None or len(subset) != len(desc.shape):
                    continue
                if not (any(str(s) in bindings
                            for s in subset.free_symbols) or _inline_indirect_idx_arrays(subset, loop_var, sdfg)):
                    continue  # a plain affine write -- the existing rules decide it
                raw = point_index_expressions(subset)
                if raw is None:
                    return None
                exprs = [substitute_indirect_bindings(x, bindings) for x in raw]
                if not all(key_map_can_read(x, sdfg, loop_var, varying) for x in exprs):
                    return None
                mask, ok = enclosing_branch_condition(state, region, bindings)
                if not ok or (mask is not None and not key_map_can_read(mask, sdfg, loop_var, varying)):
                    return None
                writes.append(JointScatterWrite(target=node.data, dim_exprs=tuple(exprs), mask_expr=mask))
    return writes


def replicate_read(sdfg: SDFG, region: LoopRegion, state: SDFGState, name: str) -> nodes.AccessNode:
    """Add a read of ``name`` to ``state``, re-establishing the binding when it is a view.

    A view carries no data of its own, so reading ``Xi_0[j]`` in a new state means re-adding the
    edge that binds ``Xi_0`` to ``Xi``; it is copied off the body state that already reads it. The
    memlet is deep-copied rather than shared -- validation rejects one Memlet object on two edges.
    """
    node = state.add_read(name)
    if not isinstance(sdfg.arrays[name], data.View):
        return node
    for body_state in region.all_states():
        for candidate in body_state.data_nodes():
            if candidate.data != name:
                continue
            edge = sdutil.get_view_edge(body_state, candidate)
            if edge is None or edge.dst is not candidate:
                continue
            src = replicate_read(sdfg, region, state, edge.src.data)
            state.add_edge(src, edge.src_conn, node, edge.dst_conn, copy.deepcopy(edge.data))
            return node
    return node


def flat_index_bound(desc: data.Array) -> symbolic.SymbolicType:
    """One past the largest flat offset ``desc`` can address, i.e. the domain its slots live in."""
    return symbolic.simplify(1 + sum((s - 1) * st for s, st in zip(desc.shape, desc.strides)))


def joint_scatter_key(plan: JointGuardPlan, loop: LoopRegion,
                      write: JointScatterWrite) -> Optional[Tuple[str, symbolic.SymbolicType]]:
    """Materialize the flat target slot ``loop`` writes on each iteration, and return it.

    The key array is what the existing one-dimensional conflict check then runs on: two entries are
    equal exactly when two iterations write the same element of ``write.target``. A masked-off
    iteration is sent to ``domain + k`` instead -- past every real slot and distinct per iteration,
    so it can collide with nothing, which is what "this iteration writes nothing" has to mean here.

    ``plan.map_dims`` names the map scopes the guard was hoisted out of. The key covers those too,
    one entry per (scope point, iteration), flattened so the conflict check still reads a single
    dimension. That widens the question from "is this loop injective" to "is the whole parallel
    space injective" -- which is what those maps' own parallelism already rests on -- and since the
    slot value carries every dimension of the target, two points differing only in a crossed map
    param hold different keys and cannot read as a collision.

    :returns: ``(key array name, exclusive bound on its values)``, or ``None`` when the loop's trip
              count is not derivable.
    """
    sdfg = plan.host_sdfg
    init = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    lstride = loop_analysis.get_loop_stride(loop)
    if init is None or end is None or lstride is None:
        return None
    trip = symbolic.simplify(symbolic.int_floor(end - init, lstride) + 1)
    desc = sdfg.arrays[write.target]
    domain = flat_index_bound(desc)

    entries = trip
    for _param, extent in plan.map_dims:
        entries = symbolic.simplify(entries * extent)
    key_name, _ = sdfg.add_array(f"_scatter_joint_key_{write.target}", [entries],
                                 dtypes.int64,
                                 transient=True,
                                 find_new_name=True)
    parent = plan.host_region
    fill_state = parent.add_state_before(plan.anchor,
                                         f"_scatter_joint_key_fill_{key_name}",
                                         is_start_block=plan.anchor is parent.start_block)

    # Every index read becomes one tasklet connector, so the body stays pure arithmetic on scalars
    # and can be a Python tasklet (a C++ one is only allowed inside a library-node expansion).
    # Keyed by (array, per-dimension index expressions): an index array is read at its own rank,
    # so a rank-2 read carries two expressions and yields a rank-2 memlet below. Flattening them
    # into one string loses the split -- ``pystr_to_symbolic`` hands back a LIST for "i, j".
    conns: Dict[Tuple[str, Tuple[str, ...]], str] = {}

    def to_connectors(expr: str) -> str:

        class Pull(ast.NodeTransformer):

            def visit_Subscript(self, node: ast.Subscript):
                self.generic_visit(node)
                if not (isinstance(node.value, ast.Name) and node.value.id in sdfg.arrays):
                    return node
                idx = node.slice
                if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
                    idx = idx.value
                dims = idx.elts if isinstance(idx, ast.Tuple) else [idx]
                key = (node.value.id, tuple(astutils.unparse(d) for d in dims))
                return ast.Name(id=conns.setdefault(key, f"__ji{len(conns)}"), ctx=ast.Load())

        return astutils.unparse(Pull().visit(ast.parse(expr, mode='eval').body))

    linear = ' + '.join(f"({to_connectors(x)}) * ({desc.strides[d]})" for d, x in enumerate(write.dim_exprs))
    # One point per (crossed map point, iteration), addressed row-major with the iteration
    # innermost. With nothing crossed this is just ``__jk``, the un-hoisted key.
    ranges = {param: f"0:{extent}" for param, extent in plan.map_dims}
    ranges['__jk'] = f"0:{trip}"
    slot = symbolic.pystr_to_symbolic('__jk')
    span = trip
    for param, extent in reversed(plan.map_dims):
        slot = slot + symbolic.pystr_to_symbolic(param) * span
        span = symbolic.simplify(span * extent)
    # A masked-off point still needs a slot of its own, past every real one.
    code = (f"__out = {linear}"
            if write.mask_expr is None else f"__out = ({linear}) if ({to_connectors(write.mask_expr)}) "
            f"else ({domain} + {symbolic.symstr(slot)})")

    # Scaffolding, not compute: flat and program-sized, so a K-dim tiling has nothing to say about
    # it. The guard's other pieces are already invisible to the vectorizer (library-node check,
    # C++ trap); this one is an ordinary Python map and needs the marker to say so.
    entry, exit_node = fill_state.add_map(f"scatter_joint_key_{key_name}{NO_VECTORIZE_MARKER}", ranges)
    tasklet = fill_state.add_tasklet(f"scatter_joint_key_{key_name}", dict.fromkeys(conns.values()), {'__out': None},
                                     code)
    # ``__jk`` counts iterations; the index expressions are written in the loop variable.
    iteration = symbolic.pystr_to_symbolic(f"({init}) + __jk * ({lstride})")
    loop_sym = symbolic.pystr_to_symbolic(loop.loop_variable)
    for (arr, idx_exprs), conn in conns.items():
        at = [symbolic.pystr_to_symbolic(e).subs({loop_sym: iteration}) for e in idx_exprs]
        fill_state.add_memlet_path(replicate_read(sdfg, loop, fill_state, arr),
                                   entry,
                                   tasklet,
                                   dst_conn=conn,
                                   memlet=mm.Memlet(data=arr, subset=subsets.Range([(a, a, 1) for a in at])))
    fill_state.add_memlet_path(tasklet,
                               exit_node,
                               fill_state.add_write(key_name),
                               src_conn='__out',
                               memlet=mm.Memlet(data=key_name, subset=subsets.Range([(slot, slot, 1)])))

    return key_name, (domain if write.mask_expr is None else symbolic.simplify(domain + entries))


def guard_joint_scatter_write(root: SDFG, loop: LoopRegion, write: JointScatterWrite, emit_trap: bool) -> Optional[str]:
    """Key ``write`` across all of its dimensions and splice a conflict check in before it runs.

    The guard sits between the key fill and the guarded block rather than at the earliest point the
    key is defined: the key is rebuilt on every pass through an enclosing loop (mandelbrot2
    recomputes it each round of its ``while``), so a guard hoisted to the definition would check a
    stale key. How far out "before it runs" is comes from :func:`joint_guard_plan`.

    :param root: the top-level SDFG the loop is somewhere inside; the plan resolves the rest.
    :returns: The duplicate-count symbol when ``emit_trap`` is False, else ``None``.
    """
    plan = joint_guard_plan(root, loop, write)
    if plan is None:
        return None
    built = joint_scatter_key(plan, loop, write)
    if built is None:
        return None
    key_name, bound = built
    parent, anchor = plan.host_region, plan.anchor
    check_state, trap_state, count_name, trap_sym = build_guard_states(plan.host_sdfg,
                                                                       key_name,
                                                                       emit_trap=emit_trap,
                                                                       region=parent,
                                                                       domain=bound)
    # ``joint_scatter_key`` already put the fill immediately before ``anchor``; splice the check
    # (and trap) into that one edge, so the order is fill -> check -> [trap] -> anchor.
    for e in list(parent.in_edges(anchor)):
        parent.remove_edge(e)
        parent.add_edge(e.src, check_state, e.data)
    if trap_state is None:
        parent.add_edge(check_state, anchor, dace.InterstateEdge(assignments={trap_sym: count_name}))
    else:
        parent.add_edge(check_state, trap_state, dace.InterstateEdge(assignments={trap_sym: count_name}))
        parent.add_edge(trap_state, anchor, dace.InterstateEdge())
    return None if emit_trap else trap_sym


def _collect_indirect_bindings(region: LoopRegion, sdfg: SDFG) -> Dict[str, Tuple[str, List[ast.AST]]]:
    """Map each symbol bound by ``region``'s interstate edges to its source data
    array plus per-dimension subscript AST nodes, when the binding is of the
    form ``sym := arr[f(loop_var)]``.

    Conservative: only a bare index-array subscript ``arr[<expr>]`` whose index
    ``<expr>`` references the loop variable is recognised (see
    :func:`_resolve_indirect_source`) -- the bare ``arr[loop_var]`` and the
    strided ``arr[c*loop_var + d]``. Non-subscript compound expressions like
    ``arr[loop_var] + 1`` are skipped; they do not arise from the DaCe Python
    frontend's scatter lowering, and extending the recognition surface risks
    misclassifying non-scatter interstate computations.
    """
    bindings: Dict[str, Tuple[str, List[ast.AST]]] = {}
    loop_var = region.loop_variable
    for e in region.all_interstate_edges():
        for lhs, rhs in (e.data.assignments or {}).items():
            resolved = _resolve_indirect_source(rhs, loop_var, sdfg)
            if resolved is None:
                resolved = resolve_staged_scalar_source(str(rhs), region, sdfg)
            if resolved is not None:
                bindings[lhs] = resolved
    return bindings


def resolve_staged_scalar_source(rhs: str, region: LoopRegion, sdfg: SDFG) -> Optional[Tuple[str, List[ast.AST]]]:
    """``(arr, dim_nodes)`` when ``rhs`` names a scalar the body filled from ``arr[f(loop_var)]``.

    A multi-dimensional scatter does not reach the interstate edge in one piece: the frontend copies
    ``Xi_0[j]`` into a scalar transient in one state and binds the symbol to that scalar on the edge
    out of it (``__sym_N__slice_0 := N__slice_0``). Reading the edge alone therefore shows a bare
    name, and the subscript that makes it a scatter sits one hop upstream -- this follows that hop.
    """
    try:
        tree = ast.parse(rhs, mode='eval').body
    except (SyntaxError, ValueError, TypeError):
        return None
    if not isinstance(tree, ast.Name):
        return None
    desc = sdfg.arrays.get(tree.id)
    if not isinstance(desc, data.Scalar):
        return None
    loop_var = region.loop_variable
    for state in region.all_states():
        for node in state.data_nodes():
            if node.data != tree.id:
                continue
            for e in state.in_edges(node):
                if e.data is None or e.data.is_empty() or not isinstance(e.src, nodes.AccessNode):
                    continue
                src_subset = e.data.src_subset if e.data.src_subset is not None else e.data.subset
                if src_subset is None or loop_var not in {str(s) for s in src_subset.free_symbols}:
                    continue
                dims = point_index_expressions(src_subset)
                if dims is None:
                    continue
                return e.src.data, [ast.parse(d, mode='eval').body for d in dims]
    return None


def _owning_sdfg(root: SDFG, loop: LoopRegion) -> SDFG:
    """Walk the SDFG tree to find the SDFG that owns ``loop``. Used so
    ``LoopToMap.apply`` reads / writes the correct arrays table on nested
    SDFGs.
    """
    for sd in root.all_sdfgs_recursive():
        if loop in list(sd.all_control_flow_regions()):
            return sd
    return root  # defensive fallback


def _hoist_slice_region(sdfg: SDFG, owner_sdfg: SDFG, index_slice: ScatterIndexSlice):
    """Return the control-flow region a sliced guard's STATES should be placed into.

    Placing them at ``owner_sdfg`` (the loop's own immediate owning sdfg) is always valid but
    turns a LoopToMap-produced single-state wrapper multi-state, which blocks a later
    ``InlineSDFG``/``InlineMultistateSDFG`` + ``MapCollapse`` pass from fusing the surrounding
    map nest (the wrapper is no longer a trivial pass-through). Hoisting past that boundary is
    safe exactly when the region hoisted to still natively defines every symbol
    ``index_slice.fixed`` references -- not merely forwards it through a NestedSDFG's
    ``symbol_mapping`` the way ``owner_sdfg`` does.

    :param sdfg: The root SDFG (searched top-down so the outermost match wins).
    :param owner_sdfg: ``loop``'s own immediate owning sdfg -- the un-hoisted fallback.
    :param index_slice: The slice whose ``fixed`` expressions must remain resolvable.
    :returns: The outermost ``LoopRegion`` whose own loop variable appears in
              ``index_slice.fixed``, or ``owner_sdfg`` when none is found.
    """
    free: Set[str] = set()
    for expr in index_slice.fixed.values():
        free |= {str(s) for s in symbolic.pystr_to_symbolic(expr).free_symbols}
    if not free:
        return owner_sdfg
    for sd in sdfg.all_sdfgs_recursive():
        for region in sd.all_control_flow_regions():
            if isinstance(region, LoopRegion) and region.loop_variable in free:
                return region
    return owner_sdfg


def _subscript_dims(idx) -> List[ast.AST]:
    """Split a subscript's (already ``ast.Index``-unwrapped) ``.slice`` into one AST node per
    dimension: the elements of an ``ast.Tuple`` for ``arr[a, b]``, else the single node itself.
    """
    return list(idx.elts) if isinstance(idx, ast.Tuple) else [idx]


def _resolve_indirect_source(rhs_str: str, loop_var: str, sdfg: SDFG) -> Optional[Tuple[str, List[ast.AST]]]:
    """Return ``(arr, dim_nodes)`` if ``rhs_str`` is ``arr[f(loop_var)]`` (``arr`` a data
    descriptor in ``sdfg`` and the index a function of ``loop_var``); ``None`` otherwise.
    ``dim_nodes`` is one AST node per subscript dimension (see :func:`_subscript_dims`).

    The index ``f(loop_var)`` may be the bare loop variable (``arr[loop_var]``,
    unit-stride scatters) or any expression referencing it (``arr[c*loop_var +
    d]``, symbolic-stride scatters such as ``ip[SSYM*i]``). Requiring the loop
    variable to appear keeps loop-invariant indices (not per-iteration scatters)
    out.
    """
    try:
        tree = ast.parse(str(rhs_str), mode='eval').body
    except (SyntaxError, ValueError, TypeError):
        return None
    if not isinstance(tree, ast.Subscript):
        return None
    if not isinstance(tree.value, ast.Name):
        return None
    arr = tree.value.id
    if arr not in sdfg.arrays:
        return None
    idx = tree.slice
    # Python <3.9 wraps the slice in ast.Index; unwrap.
    if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
        idx = idx.value
    if loop_var not in {n.id for n in ast.walk(idx) if isinstance(n, ast.Name)}:
        return None
    return arr, _subscript_dims(idx)


def _inline_indirect_idx_arrays(subset, loop_var: str, sdfg: SDFG) -> Dict[str, List[ast.AST]]:
    """Data-array names inline-subscripted inside a memlet ``subset`` with an
    index referencing ``loop_var`` -- the ``out[idx[f(i)]]`` form where the index
    array ``idx`` is embedded directly in the write subset rather than bound on an
    interstate edge.

    :param subset: A memlet subset (its ``str`` is parsed for ``idx[...]`` nodes).
    :param loop_var: The loop variable that a genuine scatter index must reference.
    :param sdfg: The SDFG whose ``arrays`` table qualifies the subscript bases.
    :returns: ``{index-array name: per-dimension subscript AST nodes}`` (empty if none).
    """
    arrays: Dict[str, List[ast.AST]] = {}
    try:
        tree = ast.parse(str(subset), mode='eval').body
    except (SyntaxError, ValueError, TypeError):
        return arrays
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)):
            continue
        arr = node.value.id
        if arr not in sdfg.arrays:
            continue
        idx = node.slice
        if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
            idx = idx.value
        if loop_var in {n.id for n in ast.walk(idx) if isinstance(n, ast.Name)}:
            arrays.setdefault(arr, _subscript_dims(idx))
    return arrays


def _nested_dynamic_scatter_idx_arrays(state, sdfg: SDFG, loop_var: str) -> dict[str, Set[str]]:
    """Integer index-array names driving a nested-SDFG data-dependent write in ``state``.

    Matches the shape a ``dace.map`` scatter (``dst[idx[i]] = ...``) lowers to: a
    ``NestedSDFG`` writes a non-transient array with a memlet whose accessed
    volume is smaller than the subset it spans (a single element scattered into a
    whole-array range). The write index lives inside the nested SDFG; this traces
    it back through the nested SDFG's input connectors to the outer integer array
    read at ``[f(loop_var)]`` and returns that array (the one whose distinctness
    the guard must check).

    :param state: The loop-body state to scan.
    :param sdfg: The SDFG owning ``state``'s arrays.
    :param loop_var: The loop variable the index read must reference.
    :returns: ``{index array name: set of arrays scattered into}`` (empty if none).
    """
    from dace.libraries.sort.nodes._helpers import is_integer_dtype

    arrays: dict[str, Set[str]] = {}
    for node in state.data_nodes():
        desc = sdfg.arrays.get(node.data)
        if desc is None or desc.transient:
            continue
        for e in state.in_edges(node):
            m = e.data
            if m is None or m.subset is None or not isinstance(e.src, nodes.NestedSDFG):
                continue
            # Data-dependent write: fewer accesses (volume) than the subset spans.
            if m.volume == m.subset.num_elements():
                continue
            # An accumulation is collision-safe by construction; the WCR may sit on either side
            # of the connector (see :func:`nested_write_is_accumulation`).
            if not indirect_write_needs_injectivity(m) or nested_write_is_accumulation(e.src, e.src_conn):
                continue
            idx_conns = _write_index_input_connectors(e.src, e.src_conn)
            for ie in state.in_edges(e.src):
                if ie.dst_conn not in idx_conns or not isinstance(ie.src, nodes.AccessNode):
                    continue
                src_desc = sdfg.arrays.get(ie.src.data)
                if (src_desc is None or src_desc.transient or not isinstance(src_desc, data.Array)
                        or not is_integer_dtype(src_desc.dtype)):
                    continue
                if ie.data is None or ie.data.subset is None:
                    continue
                if loop_var in {str(sym) for sym in ie.data.subset.free_symbols}:
                    arrays.setdefault(ie.src.data, set()).add(node.data)
    return arrays


def _write_index_input_connectors(nsdfg_node: nodes.NestedSDFG, out_conn: str) -> Set[str]:
    """Input-connector names of ``nsdfg_node`` that appear in the subset writing
    the output array ``out_conn`` inside the nested SDFG -- i.e. the connectors
    that carry the data-dependent write index.
    """
    in_conns = set(nsdfg_node.in_connectors.keys())
    idx_conns: Set[str] = set()
    for st in nsdfg_node.sdfg.all_states():
        for dn in st.data_nodes():
            if dn.data != out_conn:
                continue
            for e in st.in_edges(dn):
                if e.data is None or e.data.subset is None:
                    continue
                idx_conns |= {str(sym) for sym in e.data.subset.free_symbols if str(sym) in in_conns}
    return idx_conns


def _wrap_loop_in_dispatcher(parent, loop: LoopRegion, condition_expr: str, loop_to_map_cls) -> None:
    """Replace ``loop`` in ``parent`` with a ``ConditionalBlock`` that picks
    between a sequential clone (taken when ``condition_expr`` is true -- the
    "collision detected, fall back" branch) and a parallelised lift of the
    original loop (taken otherwise).

    The clone is a deep copy of the LoopRegion so the sequential branch keeps
    the original semantics regardless of what ``LoopToMap`` does on the other
    branch. The ConditionalBlock is spliced in at ``loop``'s former position;
    the parent's edges to/from ``loop`` are redirected to the new block.

    :param parent: The ``ControlFlowRegion`` that holds ``loop`` as one of its
        nodes. Must support ``add_node``/``remove_node``/``add_edge``.
    :param loop: The scatter loop to wrap. Must be in ``parent.nodes()``.
    :param condition_expr: The guard expression (e.g. ``"__dup_count > 0"``)
        for the ``True`` branch (sequential clone). The ``False`` branch is
        unguarded.
    :param loop_to_map_cls: ``LoopToMap`` class (injected to avoid a top-level
        import cycle through ``dace.transformation.interstate``).
    """
    import copy as _copy
    from dace.sdfg.state import ConditionalBlock, ControlFlowRegion

    if loop not in parent.nodes():
        return

    in_edges = list(parent.in_edges(loop))
    out_edges = list(parent.out_edges(loop))
    was_start = getattr(parent, 'start_block', None) is loop

    sequential_clone = _copy.deepcopy(loop)
    sequential_clone.label = loop.label + '_seq_fallback'
    # Pin the fallback so no later parallelizer re-lifts it, and so a parallelism
    # counter can treat this guarded region as fully parallel (the pinned clone is
    # the collision fallback, not a genuinely-sequential loop) -- same marker the
    # specialize-family fallbacks carry (loop_specialization.py).
    sequential_clone.pinned_sequential = True

    cb = ConditionalBlock(loop.label + '_dispatch')
    parent.add_node(cb, is_start_block=was_start, ensure_unique_name=True)  # derived label; wired by object ref

    seq_branch = ControlFlowRegion(loop.label + '_seq_branch', sdfg=parent.sdfg)
    seq_branch.add_node(sequential_clone, is_start_block=True)

    par_branch = ControlFlowRegion(loop.label + '_par_branch', sdfg=parent.sdfg)
    par_branch.add_node(loop, is_start_block=True)
    parent.remove_node(loop)

    cb.add_branch(condition_expr, seq_branch)
    cb.add_branch(None, par_branch)

    for e in in_edges:
        parent.add_edge(e.src, cb, e.data)
    for e in out_edges:
        parent.add_edge(cb, e.dst, e.data)

    # Lift the loop inside the False branch to a Map.
    owner_sdfg = parent.sdfg
    while owner_sdfg.parent_sdfg is not None:
        owner_sdfg = owner_sdfg.parent_sdfg
    instance = loop_to_map_cls()
    instance.loop = loop
    try:
        instance.apply(par_branch, owner_sdfg)
    except Exception:
        # If the lift fails on the parallel branch the sequential clone in the
        # other branch still produces the right result; codegen will compile
        # both arms unchanged.
        pass


__all__ = [
    'ScatterToGuardedMaps', 'detect_scatter_idx_arrays', 'detect_scatter_loops_and_idx_arrays',
    'indirect_write_needs_injectivity', 'nested_write_is_accumulation', 'scatter_target_arrays'
]
