# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that ensures every WCR-bearing edge sources from an :class:`AccessNode`.

The DaCe CPU codegen's WCR-resolution path emits the atomic reduction inline only
for *scalar-typed* output connectors (the typical Tasklet output shape). A
``NestedSDFG`` output is a *pointer*-typed connector, so the WCR if-branch in
:func:`dace.codegen.targets.cpu.CPUCodeGen.process_out_memlets` falls through and
emits nothing: the parallel result is wrong (approximately the last thread's
value, not the running reduction). The OMP ``reduction(...)`` clause analyser
(:func:`_collect_omp_reductions`) similarly assumes a canonical
``AccessNode -[wcr]-> MapExit`` shape upstream of the exit.

The cleanest fix without touching core codegen is to maintain the invariant that
WCR edges *always* have an :class:`AccessNode` source. This pass walks every
state in the SDFG (and every nested SDFG, recursively), finds WCR edges whose
source is a :class:`Tasklet` or :class:`NestedSDFG`, and rewrites them by
inserting a per-iteration private transient :class:`AccessNode` between the
producer and the consumer. The original WCR moves onto the new
``AccessNode -> consumer`` edge; the producer-to-AccessNode edge becomes a plain
write with no WCR.

After the pass:

* Producer (Tasklet / NestedSDFG) ``-[no wcr, memlet=_priv]->`` ``AccessNode(_priv)``
* ``AccessNode(_priv) -[wcr=op, memlet=target]->`` consumer (typically a
  :class:`MapExit`)

The downstream codegen recognises the canonical ``AccessNode``-sourced shape and
emits the correct reduction/atomic.

The pass also *seeds* the reduction slot. A WCR lowers to ``acc = acc OP val``,
which reads ``acc`` back; a fresh accumulator that is only ever WCR-written starts
uninitialized and the read-back is garbage (nondeterministic under a normal
allocator, deterministically wrong under ``MALLOC_PERTURB_``). For every WCR edge it
normalizes the pass identity-seeds the destination accumulator (0 for ``+``, 1 for
``*``, ``dtype`` bound for ``min`` / ``max``), but only when it is provably safe:

* the slot is *fresh storage* (:meth:`_is_fresh_storage`) -- a transient, or an
  out-only nested connector whose every caller binding is a transient; never a
  top-level argument or an out-only connector aliasing a live caller array
  (gramschmidt's in-place ``__tmp_78 -> &A[j]``);
* it has no plain (non-WCR) writer already initializing it, and
* the WCR writes a Map-parameter-indexed slot (:meth:`_is_write_once_wcr`), so a
  same-slot fold that continues a live prior threaded in through a differently-named
  connector (nussinov's ``_priv_table``) is left alone.

The seed is placed (:meth:`_placement`) before the innermost enclosing loop that is
summed over, else before the reduction state -- so it re-runs once per fresh
instance rather than being hoisted to the SDFG start.

The intermediate is allocated as a Scope-lifetime transient (per-Map-iteration)
matching the producer's output descriptor (dtype, shape, dtype-class). For
producers whose output connector is rank-0 / length-1, the transient is a
:class:`~dace.data.Scalar`; otherwise it is a same-shape :class:`~dace.data.Array`.

The pass is idempotent: re-running it finds no WCR edges sourced from CodeNodes
(only AccessNode-sourced WCR survives), so a fixed-point pipeline terminates
after a single iteration.
"""
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy

from dace import SDFG, SDFGState, data, dtypes
from dace.frontend.operations import detect_reduction_type
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.privatize_scatter_reduction import is_data_dependent_scatter_sink


@transformation.explicit_cf_compatible
class NormalizeWCRSource(ppl.Pass):
    """Insert an intermediate :class:`AccessNode` between every WCR-source CodeNode and its
    consumer so that all WCR-bearing edges originate at an AccessNode.
    """

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Nodes | ppl.Modifies.Memlets))

    def depends_on(self) -> Set:
        return set()

    def _output_descriptor(self, src: nodes.CodeNode, src_conn: str,
                           target_desc: Optional[data.Data]) -> Optional[data.Data]:
        """Return the data descriptor for the new private transient.

        For a :class:`NestedSDFG`, this is the inner array bound to the output connector
        (its shape + dtype are already determined by the body's IR). For a :class:`Tasklet`,
        the connector's typeclass is intentionally not consulted -- per the project rule
        "out connectors stay uninferred" -- so the descriptor is a :class:`Scalar` of the
        WCR target's element type. The value flowing through the WCR is dtype-equal to
        the accumulator slot, so this is correct for any single-element WCR write.
        """
        if isinstance(src, nodes.NestedSDFG):
            return src.sdfg.arrays.get(src_conn)
        if isinstance(src, nodes.Tasklet) and target_desc is not None:
            return data.Scalar(target_desc.dtype)
        return None

    def _make_private_desc(self, inner: data.Data) -> data.Data:
        """Build a Scope-lifetime transient descriptor matching ``inner`` element-wise.

        Length-1 / rank-0 producers get a :class:`Scalar`; multi-element producers get a
        same-shape :class:`Array`. Either way, the descriptor is fresh (deep-copied) and
        forced to ``transient=True`` + Scope lifetime so the codegen allocates it inside
        the Map body.
        """
        is_scalar = (isinstance(inner, data.Scalar) or (isinstance(inner, data.Array) and tuple(inner.shape) == (1, )))
        if is_scalar:
            return data.Scalar(inner.dtype,
                               transient=True,
                               storage=dtypes.StorageType.Default,
                               lifetime=dtypes.AllocationLifetime.Scope)
        new = copy.deepcopy(inner)
        new.transient = True
        new.storage = dtypes.StorageType.Default
        new.lifetime = dtypes.AllocationLifetime.Scope
        return new

    def _priv_subset(self, desc: data.Data) -> str:
        """Memlet subset string covering the entire private buffer."""
        if isinstance(desc, data.Scalar):
            return '0'
        return ', '.join(f'0:{s}' for s in desc.shape)

    def _enclosing_map_params(self, state: SDFGState, e, scope: Dict) -> Set[str]:
        """Parameters of every Map enclosing edge ``e`` (its source's scope) plus, when the
        edge writes through a MapExit, that Map's parameters. ``scope`` is a precomputed
        ``state.scope_dict()`` (the write-once test only inspects the pre-rewrite topology, so
        the caller computes it once before mutating the state)."""
        params: Set[str] = set()
        node = scope.get(e.src)
        while node is not None:
            if isinstance(node, nodes.MapEntry):
                params.update(node.map.params)
            node = scope.get(node)
        if isinstance(e.dst, nodes.MapExit):
            params.update(state.entry_node(e.dst).map.params)
        return params

    def _is_write_once_wcr(self, state: SDFGState, e, scope: Dict) -> bool:
        """True if this WCR writes a slot that VARIES with an enclosing (or destination) Map
        parameter -- a per-element assignment, or a per-element reduction folding a nested axis
        into ``acc[i]`` -- as opposed to a fold into a single CONSTANT slot.

        Seeding a *fresh* accumulator of this shape to the reduction identity is sound
        (``identity OP val`` is correct whether each slot is written once or a nested map folds
        into it; :meth:`_placement` handles an enclosing *loop* fold by seeding before that
        loop). A constant-subset fold (``acc[0] (max)= ...``) is refused here: such folds either
        already carry a plain identity write, or continue a live prior value threaded in through
        a differently-named connector (nussinov's ``_priv_table``, seeded by the ``table``
        input) that the freshness guard cannot see -- so they are left to their existing init.
        Freshness (vs. an aliased/live target) is enforced separately by
        :meth:`_seedable_accumulator`.
        """
        if e.data.subset is None:
            return False
        subsyms = {str(x) for x in e.data.subset.free_symbols}
        if not subsyms:
            return False
        return bool(self._enclosing_map_params(state, e, scope) & subsyms)

    def _rewrite_state(self, sdfg: SDFG, state: SDFGState) -> Tuple[int, List[Tuple[str, str, bool, SDFGState]]]:
        """Rewrite WCR edges whose source is a Tasklet or NestedSDFG.

        :returns: ``(count, seed_targets)`` where ``seed_targets`` is a list of
                  ``(accumulator_name, wcr_str, write_once, state)`` for every WCR edge
                  normalized here. The accumulator (the WCR edge's *destination* data)
                  is the reduction slot the codegen ``acc = acc OP val`` reads back; when
                  every WCR into it is ``write_once`` and it is a fresh accumulator (see
                  :meth:`_seedable_accumulator`) it starts uninitialized, so ``_apply``
                  seeds it to the reduction identity. The freshly-minted ``_wcr_priv``
                  buffer is *not* the accumulator -- it is plain-written by ``src`` -- so
                  it needs no seed.
        """
        rewritten = 0
        seed_targets: List[Tuple[str, str, bool, SDFGState]] = []
        # Snapshot first; we mutate the edge set inside the loop.
        targets = [
            e for e in state.edges()
            if e.data is not None and e.data.wcr is not None and isinstance(e.src, (nodes.Tasklet, nodes.NestedSDFG))
        ]
        # The write-once test inspects the pre-rewrite topology, and inserting the private
        # AccessNode below invalidates the scope-dict cache; compute it once, up front.
        scope = state.scope_dict() if targets else {}
        write_once = {e: self._is_write_once_wcr(state, e, scope) for e in targets}
        for e in targets:
            src = e.src
            src_conn = e.src_conn
            if src_conn is None:
                continue
            # InOut connector on a NestedSDFG: splitting only the OUT side onto
            # ``_wcr_priv_<src>_<conn>`` would break the InOut invariant
            # (validation rejects an in/out pair on the same connector name
            # pointing at two different external arrays). Skip the rewrite for
            # such edges -- the WCR stays on the direct NestedSDFG-output edge
            # and codegen falls back to its atomic-add path.
            if isinstance(src, nodes.NestedSDFG) and src_conn in src.in_connectors:
                continue
            # Data-dependent scatter reduction (azimint histogram): the accumulation runs
            # on the inner single-element WCR straight into the accumulator, and any WCR on
            # this NestedSDFG->MapExit edge is only the reduction-clause marker. Wrapping it
            # in a per-iteration whole-array ``_wcr_priv`` buffer would (a) undo that and
            # (b) be unsound -- only one element of the buffer is written per iteration, the
            # rest read back uninitialised. Skip ANY such scatter sink, REGARDLESS of op:
            # ``PrivatizeScatterReduction`` owns the OpenMP-reducible ones (leaving the WCR
            # on the direct edge, codegen emits nothing for it), and a non-reducible op
            # (``-`` / ``/``) that neither that pass nor ``NormalizeWCR`` rewrote must fall
            # back to the correct per-element atomic rather than a broken whole-buffer copy.
            if isinstance(src, nodes.NestedSDFG) and is_data_dependent_scatter_sink(src, src_conn):
                continue
            target_desc = sdfg.arrays.get(e.data.data) if e.data.data else None
            inner = self._output_descriptor(src, src_conn, target_desc)
            if inner is None:
                continue
            priv_desc = self._make_private_desc(inner)
            # Skip when the private buffer's shape depends on symbols not defined at this
            # SDFG's scope -- e.g. a triangular reduction whose output extent is ``i+1``
            # in an enclosing loop variable. Materializing the buffer (and its memlets) at
            # this scope would leak ``i`` as a free symbol. Leave the WCR on the direct
            # edge so codegen falls back to its atomic-add path.
            if {str(s) for s in priv_desc.free_symbols} - set(sdfg.symbols.keys()):
                continue
            priv_name = sdfg.add_datadesc(f'_wcr_priv_{src.label}_{src_conn}', priv_desc, find_new_name=True)
            priv_node = state.add_access(priv_name)

            state.add_edge(src, src_conn, priv_node, None, Memlet(data=priv_name, subset=self._priv_subset(priv_desc)))
            state.add_edge(priv_node, None, e.dst, e.dst_conn, copy.deepcopy(e.data))
            state.remove_edge(e)
            rewritten += 1
            if e.data.data:
                seed_targets.append((e.data.data, e.data.wcr, write_once[e], state))
        return rewritten, seed_targets

    def _plain_written(self, sd: SDFG) -> Set[str]:
        """Every data name in ``sd`` that has a plain (non-WCR) writer -- an edge into an
        AccessNode of that name with ``wcr is None``. Such a name is already initialized, so
        the reduction seeds off it and must not be re-seeded. Keyed on ``e.dst.data`` (NOT
        ``e.data.data``): a source-oriented copy ``read(B) -> write(acc)`` carries the memlet
        as ``B[...]`` (``e.data.data == 'B'``) yet still plain-initializes ``acc``. Computed
        once per SDFG (not per accumulator)."""
        out: Set[str] = set()
        for st in sd.all_states():
            for e in st.edges():
                if (e.data is not None and e.data.wcr is None and not e.data.is_empty()
                        and isinstance(e.dst, nodes.AccessNode)):
                    out.add(e.dst.data)
        return out

    def _is_fresh_storage(self, sd: SDFG, name: str, desc: data.Data) -> bool:
        """``name`` is DaCe-owned fresh scratch, safe to identity-seed: a transient, or a
        write-only nested-SDFG output connector whose EVERY caller binding is to a transient.

        A top-level non-transient array is a program argument (the caller owns its value; a
        bare ``C += A@B`` folds onto caller data). A nested in+out (or in-only) connector
        carries a caller value. Critically, an out-only connector can still ALIAS a live caller
        array -- gramschmidt binds the ``__tmp_78`` output to ``&A[j]`` and accumulates
        ``A[:,j] += -Q*R`` in place -- so it is fresh only when every caller edge binds it to a
        transient (a ``_wcr_priv`` scratch), never a non-transient array. (Seeding an aliased
        live array would erase it.)"""
        if desc.transient:
            return True
        parent = sd.parent_nsdfg_node
        if parent is None:
            return False  # top-level non-transient argument -> caller owns it
        if name in parent.in_connectors or name not in parent.out_connectors:
            return False
        parent_state, parent_sdfg = sd.parent, sd.parent_sdfg
        if parent_state is None or parent_sdfg is None:
            return False
        bound = [oe for oe in parent_state.out_edges(parent) if oe.src_conn == name]
        if not bound:
            return False
        return all((cdesc := parent_sdfg.arrays.get(oe.data.data)) is not None and cdesc.transient for oe in bound)

    def _seedable_accumulator(self, sd: SDFG, name: str, plain_written: Set[str]) -> bool:
        """True if ``name`` is a fresh WCR accumulator safe to identity-seed: fresh storage
        (:meth:`_is_fresh_storage`), with no existing plain writer and no shape symbols undefined
        at this SDFG's scope (the fill map would otherwise leak a free symbol). :meth:`_placement`
        keeps the seed inside the accumulator's own scope, so a mis-scoped seed would fail
        validation rather than silently corrupt."""
        desc = sd.arrays.get(name)
        if desc is None:
            return False
        if name in plain_written:
            return False
        if {str(s) for s in desc.free_symbols} - set(sd.symbols.keys()):
            return False
        return self._is_fresh_storage(sd, name, desc)

    def _enclosing_loops(self, block) -> List[LoopRegion]:
        """The LoopRegion ancestors of ``block``, innermost first."""
        loops: List[LoopRegion] = []
        cur = block.parent_graph
        while cur is not None:
            if isinstance(cur, LoopRegion):
                loops.append(cur)
            cur = cur.parent_graph if isinstance(cur, ControlFlowRegion) else None
        return loops

    def _placement(self, sd: SDFG, name: str, states: Set[SDFGState]):
        """The block to prepend the seed before, so the reduction runs once per fresh instance:
        immediately before the innermost enclosing loop that is *summed over* (its variable does
        not index ``name``), else before the WCR-writing state itself (a parallel-map reduction,
        or a per-element write, re-seeded by any enclosing loop it sits inside). ``None`` when
        the placement is ambiguous -- multiple WCR-writing states, or two summed loops of
        unknown extent -- in which case the accumulator is left unseeded rather than mis-seeded."""
        if len(states) != 1:
            return None
        state = next(iter(states))
        idx_syms: Set[str] = set()
        for e in state.edges():
            if (e.data is not None and e.data.data == name and e.data.wcr is not None
                    and isinstance(e.dst, nodes.AccessNode) and e.dst.data == name):
                sub = e.data.get_dst_subset(e, state) or e.data.subset
                if sub is not None:
                    idx_syms |= {str(s) for s in sub.free_symbols}
        summed = [
            ln for ln in self._enclosing_loops(state) if ln.loop_variable and str(ln.loop_variable) not in idx_syms
        ]
        if not summed:
            return state
        return summed[0] if len(summed) == 1 else None

    def _identity_literal(self, val: Any, dtype: dtypes.typeclass) -> str:
        """Codegen-safe tasklet literal for the reduction identity at ``dtype``. Converts the
        numpy scalar to a *Python* scalar first: ``repr(numpy.float64(0.0))`` is
        ``'np.float64(0.0)'`` under NumPy 2.x, which codegen leaves as an undeclared ``np`` --
        ``float`` / ``int`` / ``bool`` / ``complex`` round-trip to plain literals."""
        nptype = dtype.type
        if numpy.issubdtype(nptype, numpy.bool_):
            return 'True' if val else 'False'
        if numpy.issubdtype(nptype, numpy.complexfloating):
            c = complex(val)
            return f'complex({c.real!r}, {c.imag!r})'
        if numpy.issubdtype(nptype, numpy.integer):
            return str(int(val))
        return repr(float(val))

    def _seed_accumulator(self, sd: SDFG, name: str, wcr: str, states: Set[SDFGState]) -> bool:
        """Prepend an identity-seed state so the fresh accumulator starts defined before its
        reduction. The seed is an explicit identity-init *tasklet* (not ``setzero``): a
        bitwise-zero fill is wrong for ``*`` (identity 1) and ``min`` / ``max``. Returns True
        when a seed was emitted."""
        desc = sd.arrays.get(name)
        if desc is None:
            return False
        red = detect_reduction_type(wcr)
        if red is None:
            return False
        try:
            val = dtypes.reduction_identity(desc.dtype, red)
        except (TypeError, ValueError):
            return False  # e.g. min/max on a dtype with no defined bound (complex)
        if val is None:
            return False
        before = self._placement(sd, name, states)
        if before is None:
            return False
        parent: ControlFlowRegion = before.parent_graph
        seed = parent.add_state_before(before, label=name + '_wcr_seed', is_start_block=parent.start_block is before)
        lit = self._identity_literal(val, desc.dtype)
        w = seed.add_write(name)
        if isinstance(desc, data.Scalar) or tuple(desc.shape) == (1, ):
            t = seed.add_tasklet(name + '_wcr_seed', {}, {'__out'}, f'__out = {lit}')
            seed.add_edge(t, '__out', w, None, Memlet(data=name, subset='0'))
        else:
            me, mx = seed.add_map(name + '_wcr_seed', {f'_wcrseed_i{d}': f'0:{s}' for d, s in enumerate(desc.shape)})
            t = seed.add_tasklet(name + '_wcr_seed', {}, {'__out'}, f'__out = {lit}')
            idx = ', '.join(f'_wcrseed_i{d}' for d in range(len(desc.shape)))
            seed.add_edge(me, None, t, None, Memlet())
            seed.add_memlet_path(t, mx, w, src_conn='__out', memlet=Memlet(data=name, subset=idx))
        return True

    def _apply(self, sdfg: SDFG) -> int:
        total = 0
        # Per SDFG: accumulator name -> (wcr, all_write_once, {writing states}). Deduped so a
        # slot written by several normalized WCR edges is seeded once and only when *every* such
        # write is write-once (a lone same-slot fold disqualifies it). Collected across the whole
        # traversal, then seeded afterwards so inserting seed states does not perturb iteration.
        seed_reqs: Dict[int, Tuple[SDFG, Dict[str, list]]] = {}
        for sd in sdfg.all_sdfgs_recursive():
            for state in list(sd.all_states()):
                n, reqs = self._rewrite_state(sd, state)
                total += n
                if reqs:
                    _, names = seed_reqs.setdefault(id(sd), (sd, {}))
                    for name, wcr, write_once, st in reqs:
                        entry = names.setdefault(name, [wcr, True, set()])
                        entry[1] = entry[1] and write_once
                        entry[2].add(st)
        for sd, names in seed_reqs.values():
            plain_written = self._plain_written(sd)  # one scan per SDFG, reused for every accumulator
            for name, (wcr, all_write_once, states) in names.items():
                if all_write_once and self._seedable_accumulator(sd, name, plain_written):
                    self._seed_accumulator(sd, name, wcr, states)
        return total

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """Rewrite every WCR-bearing edge so its source is an :class:`AccessNode`.

        :param sdfg: The SDFG to normalize.
        :param pipeline_results: Results of prior passes in the pipeline (unused).
        :returns: ``None`` if no edges were rewritten; otherwise a single-entry dict
                  with the rewritten count under key ``normalized_wcr_edges``.
        """
        n = self._apply(sdfg)
        if n == 0:
            return None
        sdfg.validate()
        return {'normalized_wcr_edges': {str(n)}}
