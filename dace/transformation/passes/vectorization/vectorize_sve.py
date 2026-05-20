# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""SVE-style finalize orchestrator (the ``sve_style='fixed'`` chain).

A single coordinating :class:`Pass` (the M3.2/M3.3 pattern — composition
order and arguments depend on runtime SDFG state, so a flat declarative
pipeline cannot express it). It owns the whole SVE-style finalize and,
critically, **captures the global trip bound once at tile time** (the
single source of truth) and threads it explicitly to the mask + Min-swap
passes, immune to any intervening pass mutating the ``core`` map range.

Chain (analyze-clean-then-Min):

1. For each eligible innermost single-param step-1 map: capture
   ``global_ub = ub + 1`` (the original exclusive trip bound) *now*,
   compute the clean block ``B = roundup(ceil(trip / num_cores), W)``,
   and ``MapTiling(divides_evenly=True)`` it into a ``core`` outer map
   plus a divisible per-core block. ``divides_evenly=True`` keeps every
   map range affine so downstream divisibility / subset analysis never
   sees a ``Min`` (avoids the ``SympifyError`` a ``Min`` in a map range
   triggers).
2. ``NestInnermostMapBodyIntoNSDFG(nest_provably_divisible=True)`` — the
   per-core block is divisible by design but still needs a NestedSDFG
   body for the mask.
3. ``GenerateIterationMask(mode='global', global_ub=<captured>)`` — the
   global-keyed ``_iter_mask`` (``i + l < global_ub``).
4. The shared ``Vectorize`` pass W-strides the divisible block and emits
   the masked (``_av_masked``) variants.
5. ``Detect{Gather,Scatter}`` (and the strided detectors when
   ``lower_to_intrinsics``) collapse per-lane fans to masked intrinsics
   — mandatory under masking (a per-lane scalar fan faults on inactive
   lanes).
6. ``MapToForLoop`` turns each W-strided per-core map into a
   :class:`LoopRegion`.
7. ``ForLoopToMaskedWhile(global_ub=<captured>)`` Min-swaps the loop
   condition and W-stride-normalizes the update; it re-derives the bound
   from the ``core`` map and **asserts it equals the captured value**
   (loud failure if any pass perturbed the range).

The bound is *captured at step 1*, where the original map is in hand,
and used in steps 3 and 7 — never re-derived from a secondary artifact
several passes later.
"""
from typing import List, Optional, Tuple

import dace
from dace import properties
from dace import symbolic
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.dataflow.map_expansion import MapExpansion
from dace.transformation.passes.vectorization.nest_innermost_map_body import NestInnermostMapBodyIntoNSDFG
from dace.transformation.passes.vectorization.generate_iteration_mask import GenerateIterationMask
from dace.transformation.passes.vectorization.for_loop_to_masked_while import ForLoopToMaskedWhile
from dace.transformation.passes.vectorization.detect_gather import DetectGather
from dace.transformation.passes.vectorization.detect_scatter import DetectScatter
from dace.transformation.passes.vectorization.detect_strided_load import DetectStridedLoad
from dace.transformation.passes.vectorization.detect_strided_store import DetectStridedStore
from dace.transformation.passes.vectorization.detect_multi_dim_strided_load import DetectMultiDimStridedLoad
from dace.transformation.passes.vectorization.detect_multi_dim_strided_store import DetectMultiDimStridedStore
from dace.transformation.passes.vectorization.remove_vector_maps import RemoveVectorMaps
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map

_CORE_PREFIX = "core"


@properties.make_properties
class SveStyleFinalize(ppl.Pass):
    """Drive the ``sve_style='fixed'`` chain with a tile-time-captured bound."""

    CATEGORY: str = "Vectorization Preparation"

    def __init__(self,
                 vectorizer,
                 vector_width: int,
                 num_cores: int,
                 lower_to_intrinsics: bool = True,
                 eliminate_trivial_vector_map: bool = True):
        super().__init__()
        self._vectorizer = vectorizer
        self._W = vector_width
        self._P = num_cores
        self._lower_to_intrinsics = lower_to_intrinsics
        self._eliminate_trivial_vector_map = eliminate_trivial_vector_map

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    @staticmethod
    def _lane_winner_param(map_entry: dace.nodes.MapEntry, state: dace.SDFGState) -> str:
        """Pick the map param that most often drives the unit-stride dim
        of an array access in this map's scope. That param is the best
        lane dim (W-stride across it touches contiguous memory); it
        should end up innermost so the vectorizer's W-strip-mining
        operates on the contiguous axis. Falls back to the last param
        when no analysis info is available (default-correct for
        C-layout where the last dim is already the contiguous one).

        :param map_entry: The multi-param map to analyze.
        :param state: The state containing ``map_entry``.
        :returns: The param name with the highest unit-stride-dim
            access count, or the last param as a safe fallback.
        """
        from collections import Counter
        counts: Counter = Counter()
        scope = state.scope_subgraph(map_entry)
        for node in scope.nodes():
            for e in list(scope.in_edges(node)) + list(scope.out_edges(node)):
                if e.data is None or e.data.data is None:
                    continue
                if e.data.data not in state.sdfg.arrays:
                    continue
                arr = state.sdfg.arrays[e.data.data]
                strides = getattr(arr, "strides", None)
                if not strides:
                    continue
                try:
                    unit_dim = [str(s) for s in strides].index("1")
                except ValueError:
                    continue
                if e.data.subset is None or unit_dim >= len(e.data.subset):
                    continue
                dim_expr = e.data.subset[unit_dim]
                begin = dim_expr[0] if isinstance(dim_expr, tuple) else dim_expr
                try:
                    free = {str(s) for s in begin.free_symbols}
                except Exception:
                    continue
                for p in map_entry.map.params:
                    if p in free:
                        counts[p] += 1
        if not counts:
            return map_entry.map.params[-1]
        return counts.most_common(1)[0][0]

    @staticmethod
    def _permute_winner_last(map_entry: dace.nodes.MapEntry, winner: str) -> bool:
        """Rearrange ``map_entry.params`` and ``range.ranges`` so
        ``winner`` is the last param (innermost after MapExpansion).
        In-place; semantically equivalent (the cartesian product is
        unchanged). Returns True iff a swap occurred.

        :param map_entry: Map whose params to permute.
        :param winner: Param to move to the last position.
        :returns: True if the order changed; False if winner was
            already last or absent.
        """
        params = list(map_entry.map.params)
        if winner not in params or params[-1] == winner:
            return False
        ranges = list(map_entry.map.range.ranges)
        idx = params.index(winner)
        params.append(params.pop(idx))
        ranges.append(ranges.pop(idx))
        map_entry.map.params = params
        map_entry.map.range = dace.subsets.Range(ranges)
        return True

    def _expand_multi_to_1d(self, sdfg: dace.SDFG) -> int:
        """Analyze-permute-expand every multi-param innermost map so the
        SVE chain sees only 1D maps. For each multi-param map: pick the
        lane winner (most unit-stride accesses), permute it to last,
        ``MapExpansion`` splits into nested 1D maps (the deepest is the
        winner). Mark every post-expansion deepest 1D map nested under
        another MapEntry ``Sequential`` — these come from multi-dim
        expansion and don't need ``num_cores`` core-tiling (the outer
        expanded maps already provide parallelism).

        :param sdfg: SDFG to transform in place.
        :returns: Number of multi-param maps expanded.
        """
        multi_targets = [
            (n, g) for n, g in list(sdfg.all_nodes_recursive())
            if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState) and is_innermost_map(g, n)
            and len(n.map.params) > 1 and not any(p.startswith(_CORE_PREFIX) for p in n.map.params)
        ]
        expanded = 0
        for n, g in multi_targets:
            winner = self._lane_winner_param(n, g)
            self._permute_winner_last(n, winner)
            MapExpansion.apply_to(g.sdfg, verify=True, save=False, map_entry=n)
            expanded += 1
        if expanded:
            # Mark deepest post-expansion 1D maps Sequential (those
            # nested under another MapEntry — i.e. came from an expanded
            # multi-param). True top-level 1D maps are unchanged.
            for n, g in list(sdfg.all_nodes_recursive()):
                if not (isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState)):
                    continue
                if not is_innermost_map(g, n) or len(n.map.params) != 1:
                    continue
                if isinstance(g.scope_dict().get(n), dace.nodes.MapEntry):
                    n.map.schedule = dace.dtypes.ScheduleType.Sequential
        return expanded

    def _eligible_innermost_maps(self,
                                 sdfg: dace.SDFG) -> List[Tuple[dace.nodes.MapEntry, dace.SDFGState, str, object]]:
        """Collect ``(map_entry, state, global_ub, block)`` for every
        tileable innermost map, capturing the bound *before* tiling.

        :param sdfg: The SDFG to scan.
        :returns: One tuple per eligible single-param step-1 innermost map.
        """
        W, P = self._W, self._P
        out = []
        for n, g in list(sdfg.all_nodes_recursive()):
            if not (isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState)):
                continue
            if not is_innermost_map(g, n):
                continue
            if len(n.map.params) != 1:
                continue
            if any(p.startswith(_CORE_PREFIX) for p in n.map.params):
                continue
            lb, ub, step = n.map.range[-1]
            if (step != 1) and (str(step) != "1"):
                continue
            trip = symbolic.simplify(ub - lb + 1)
            global_ub = str(ub + 1)  # original exclusive bound, captured now
            block = symbolic.int_ceil(symbolic.int_ceil(trip, P), W) * W
            out.append((n, g, global_ub, block))
        return out

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Run the SVE-style finalize. Returns 1 if it fired, else ``None``.

        :param sdfg: The SDFG to transform in place.
        :param _: Unused pipeline results.
        :raises NotImplementedError: if eligible innermost maps have
            distinct global trips (first-cut supports one global bound).
        """
        W = self._W
        # 0. Multi-dim handling: permute lane winner to last + MapExpansion
        #    + mark expanded inner maps Sequential (skip core-tile for them).
        self._expand_multi_to_1d(sdfg)

        targets = self._eligible_innermost_maps(sdfg)
        if not targets:
            return None
        gubs = {gub for _, _, gub, _ in targets}
        if len(gubs) != 1:
            raise NotImplementedError(f"sve_style: eligible innermost maps have distinct global trips {sorted(gubs)}; "
                                      f"the first-cut SVE chain threads a single captured global_ub. Split the kernel "
                                      f"or restrict via apply_on_maps.")
        global_ub = gubs.pop()

        # 1. Tile only TRUE 1D maps (no MapEntry scope-parent) into
        #    a clean divisible per-core block. Multi-dim maps from
        #    expansion are skipped: their outer expanded maps already
        #    provide parallelism, the inner is Sequential — no need
        #    for num_cores partitioning on the inner.
        for n, g, _gub, block in targets:
            if isinstance(g.scope_dict().get(n), dace.nodes.MapEntry):
                continue  # multi-dim inner — skip core-tile
            MapTiling.apply_to(g.sdfg,
                               options={
                                   "tile_sizes": (block, ),
                                   "prefix": _CORE_PREFIX,
                                   "divides_evenly": True,
                                   "tile_trivial": True,
                               },
                               verify=True,
                               save=False,
                               map_entry=n)

        # 2-3. Nest the divisible block body + attach the global mask.
        NestInnermostMapBodyIntoNSDFG(vector_width=W, nest_provably_divisible=True).apply_pass(sdfg, {})
        GenerateIterationMask(vector_width=W, mode="global", global_ub=global_ub).apply_pass(sdfg, {})

        # 4. Vectorize the divisible block (emits the masked variants).
        self._vectorizer.apply_pass(sdfg, {})

        # 5. Collapse per-lane gather/scatter (and strided) fans to masked
        #    intrinsics — mandatory under masking.
        DetectGather().apply_pass(sdfg, {})
        DetectScatter().apply_pass(sdfg, {})
        if self._lower_to_intrinsics:
            DetectStridedLoad().apply_pass(sdfg, {})
            DetectStridedStore().apply_pass(sdfg, {})
            DetectMultiDimStridedLoad().apply_pass(sdfg, {})
            DetectMultiDimStridedStore().apply_pass(sdfg, {})

        # 6. Each W-strided per-core map -> a LoopRegion.
        for n, g in list(sdfg.all_nodes_recursive()):
            if not (isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState)):
                continue
            if any(p.startswith(_CORE_PREFIX) for p in n.map.params):
                continue
            st = n.map.range[-1][2]
            if (st == W) or (str(st) == str(W)):
                MapToForLoop.apply_to(g.sdfg, verify=True, save=False, map_entry=n)

        # 7. for-loop -> masked while (Min-swap + tile-time-bound assert).
        ForLoopToMaskedWhile(vector_width=W, global_ub=global_ub).apply_pass(sdfg, {})

        if self._eliminate_trivial_vector_map:
            RemoveVectorMaps().apply_pass(sdfg, {})
        return 1


# -------------------------------------------------------------------------
# sve_style='variable' (true runtime VL via svwhilelt_b64 + svcntd)
# -------------------------------------------------------------------------
# Replaces each recognised innermost map body with a CPP tasklet whose
# body is the SVE while-loop pattern, guarded behind ``#if defined(
# __ARM_FEATURE_SVE)`` with a scalar fallback so the SDFG compiles on
# x86 (the fallback executes; the SVE branch is taken on SVE hosts).
# This first cut recognises simple per-element arithmetic on float64
# arrays: ``out[i] = a[i] (op) b[i]`` for op in {+, -, *, /} and
# ``out[i] = a[i]`` (copy). Unsupported map shapes raise
# ``NotImplementedError`` so failures are loud rather than silent.

_SVE_OP_INTRINSIC = {
    "+": "svadd_f64_z",
    "-": "svsub_f64_z",
    "*": "svmul_f64_z",
    "/": "svdiv_f64_z",
}


@properties.make_properties
class SveStyleVariableFinalize(ppl.Pass):
    """Replace recognised innermost float64 element-wise maps with a CPP
    tasklet emitting the SVE runtime-VL while-loop. Unsupported shapes
    raise ``NotImplementedError`` (loud failure, no silent fallback)."""

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    @staticmethod
    def _classify_chain_body(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
        """Recognise a linear chain of single-binop tasklets (each
        ``__out = (__in1 OP __in2)``) ending in an assign tasklet that
        writes to the outer destination. Each binop consumes either
        outer arrays or the previous binop's transient output (via an
        intermediate access node). Handles axpy, triad
        (``d = a + b + c``), and longer associative chains uniformly.

        :returns: ``(steps, out_array, global_ub_str)`` where ``steps``
            is a list ``[(op, outer_inputs_or_None, n_outer)]`` describing
            the chain — the emitter walks it building nested SVE
            intrinsics. Returns ``None`` if not a recognised chain.
        """
        if len(map_entry.map.params) != 1:
            return None
        lb, ub, step = map_entry.map.range[-1]
        if (step != 1) and (str(step) != "1"):
            return None
        sdfg = state.sdfg
        map_exit = state.exit_node(map_entry)
        body_nodes = [
            n for n in state.all_nodes_between(map_entry, map_exit)
            if not isinstance(n, (dace.nodes.MapEntry, dace.nodes.MapExit))
        ]
        tasklets = [n for n in body_nodes if isinstance(n, dace.nodes.Tasklet)]
        if not tasklets:
            return None
        import re
        binop_re = re.compile(r"^(\w+)\s*=\s*\(?\s*(\w+)\s*([+\-*/])\s*(\w+)\s*\)?\s*$")
        copy_re = re.compile(r"^(\w+)\s*=\s*(\w+)\s*$")

        # Classify each tasklet as binop, copy, or unknown.
        info = {}
        for tk in tasklets:
            code = tk.code.as_string.strip().rstrip(";").strip()
            mb = binop_re.match(code)
            mc = copy_re.match(code)
            if mb:
                info[tk] = ("binop", mb.groups())
            elif mc:
                info[tk] = ("copy", mc.groups())
            else:
                return None  # tasklet doesn't fit the single-op contract
        # Topological order — each binop's outputs flow forward.
        from collections import deque
        order = []
        in_deg = {tk: sum(1 for ie in state.in_edges(tk) if isinstance(ie.src, dace.nodes.AccessNode)
                          and ie.src in body_nodes) for tk in tasklets}
        q = deque(tk for tk in tasklets if in_deg[tk] == 0)
        while q:
            tk = q.popleft()
            order.append(tk)
            for oe in state.out_edges(tk):
                # Out-edge -> AccessNode -> next Tasklet
                if isinstance(oe.dst, dace.nodes.AccessNode):
                    for ne in state.out_edges(oe.dst):
                        if isinstance(ne.dst, dace.nodes.Tasklet) and ne.dst in in_deg:
                            in_deg[ne.dst] -= 1
                            if in_deg[ne.dst] == 0:
                                q.append(ne.dst)
        if len(order) != len(tasklets):
            return None  # not a DAG / has cycles or unexpected structure

        # Split into compute chain (binops or single copy) + optional
        # trailing assign (copy) that writes to the outer destination.
        compute = [tk for tk in order if info[tk][0] == "binop"]
        # The final tasklet feeding the map exit should be either:
        # - the last binop (if its output goes directly to the outer dst), or
        # - a trailing assign/copy tasklet right after the last binop.
        # Trace the chain: walk from each binop's output through intermediate
        # access nodes; identify the outer source for each binop's "fresh" input.
        if not compute:
            # Pure copy kernel.
            if len(tasklets) != 1 or info[tasklets[0]][0] != "copy":
                return None
            tk = tasklets[0]
            out_conn, a_conn = info[tk][1]

            def _outer_in(conn):
                for ie in state.in_edges(tk):
                    if ie.dst_conn != conn or ie.data is None:
                        continue
                    for hop in reversed(state.memlet_path(ie)):
                        if isinstance(hop.src, dace.nodes.AccessNode):
                            return hop.src.data
                return None

            def _outer_out(conn):
                for oe in state.out_edges(tk):
                    if oe.src_conn != conn or oe.data is None:
                        continue
                    for hop in state.memlet_path(oe):
                        if isinstance(hop.dst, dace.nodes.AccessNode):
                            return hop.dst.data
                return None

            a_arr = _outer_in(a_conn)
            d_arr = _outer_out(out_conn)
            if a_arr is None or d_arr is None:
                return None
            if sdfg.arrays[a_arr].dtype != dace.float64 or sdfg.arrays[d_arr].dtype != dace.float64:
                return None
            if len(sdfg.arrays[a_arr].shape) != 1 or len(sdfg.arrays[d_arr].shape) != 1:
                return None
            return ([("copy", [a_arr])], d_arr, str(ub + 1))

        # Build the binop chain. For each binop in order, identify
        # which of its 2 inputs is the prev-binop's output (transient,
        # not directly an outer source) vs an outer-source array.
        steps = []
        prev_out_access_data = None  # data name of the intermediate access node from prev binop

        def _input_array_via_path(tk, conn):
            for ie in state.in_edges(tk):
                if ie.dst_conn != conn or ie.data is None:
                    continue
                for hop in reversed(state.memlet_path(ie)):
                    if isinstance(hop.src, dace.nodes.AccessNode):
                        return hop.src.data
            return None

        for tk in compute:
            out_conn, l_conn, op, r_conn = info[tk][1]
            if op not in _SVE_OP_INTRINSIC:
                return None
            l_data = _input_array_via_path(tk, l_conn)
            r_data = _input_array_via_path(tk, r_conn)
            if l_data is None or r_data is None:
                return None
            # Identify which input is the previous-result (transient).
            outer_inputs = []
            uses_prev = False
            for d in (l_data, r_data):
                if prev_out_access_data is not None and d == prev_out_access_data:
                    uses_prev = True
                else:
                    outer_inputs.append(d)
            if prev_out_access_data is not None and not uses_prev:
                return None  # broken chain (not linear)
            steps.append((op, outer_inputs))
            # Record this binop's output (transient access node it writes to).
            for oe in state.out_edges(tk):
                if oe.src_conn == out_conn and isinstance(oe.dst, dace.nodes.AccessNode):
                    prev_out_access_data = oe.dst.data
                    break

        # Find the outer destination: trace from the last binop's output
        # forward through any assign tasklet to the outer destination AccessNode.
        out_array = None
        last_tk = compute[-1]
        for oe in state.out_edges(last_tk):
            if oe.data is None:
                continue
            cur = oe
            for _ in range(10):  # bounded walk
                hop_dst = None
                for hop in state.memlet_path(cur):
                    if isinstance(hop.dst, dace.nodes.AccessNode):
                        hop_dst = hop.dst
                        break
                if hop_dst is None:
                    break
                # Outer destination AccessNodes are those NOT in the
                # body_nodes (they live outside the map scope).
                if hop_dst not in body_nodes:
                    out_array = hop_dst.data
                    break
                # Otherwise it's a transient; advance through next tasklet.
                next_t = None
                for ne in state.out_edges(hop_dst):
                    if isinstance(ne.dst, dace.nodes.Tasklet):
                        next_t = ne.dst
                        break
                if next_t is None:
                    out_array = hop_dst.data
                    break
                # Follow the assign tasklet's first output edge.
                next_outs = state.out_edges(next_t)
                if not next_outs:
                    out_array = hop_dst.data
                    break
                cur = next_outs[0]
            if out_array is not None:
                break
        if out_array is None:
            return None
        # All arrays float64 1D.
        all_arrs = [arr for (_, outer_ins) in steps for arr in outer_ins] + [out_array]
        for nm in all_arrs:
            arr = sdfg.arrays.get(nm)
            if arr is None or arr.dtype != dace.float64 or len(arr.shape) != 1:
                return None
        return (steps, out_array, str(ub + 1))

    @staticmethod
    def _classify_simple_axpy_body(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
        """Recognise ``out[i] = a[i] (op) b[i]`` or ``out[i] = a[i]``
        as the body of a single-param step-1 1D map.

        :returns: ``(op, in_names, out_name, dtype)`` or ``None`` if the
            body is not a recognised simple axpy.
        """
        if len(map_entry.map.params) != 1:
            return None
        lb, ub, step = map_entry.map.range[-1]
        if (step != 1) and (str(step) != "1"):
            return None
        body_nodes = [
            n for n in state.all_nodes_between(map_entry, state.exit_node(map_entry))
            if not isinstance(n, (dace.nodes.MapEntry, dace.nodes.MapExit))
        ]
        tasklets = [n for n in body_nodes if isinstance(n, dace.nodes.Tasklet)]
        if not tasklets:
            return None
        sdfg = state.sdfg
        import re
        # Find the compute tasklet: ``OUT = (IN1 OP IN2)`` (parens
        # optional). The dace frontend's canonical shape for ``c[i] =
        # a[i] + b[i]`` is a compute tasklet + an assign tasklet + an
        # intermediate access node; we pick the compute one.
        binop_re = re.compile(r"^(\w+)\s*=\s*\(?\s*(\w+)\s*([+\-*/])\s*(\w+)\s*\)?\s*$")
        copy_re = re.compile(r"^(\w+)\s*=\s*(\w+)\s*$")
        compute = None
        for tk in tasklets:
            code = tk.code.as_string.strip().rstrip(";").strip()
            m = binop_re.match(code)
            if m:
                compute = (tk, "binop", m.groups())
                break
        if compute is None and len(tasklets) == 1:
            m = copy_re.match(tasklets[0].code.as_string.strip().rstrip(";").strip())
            if m:
                compute = (tasklets[0], "copy", m.groups())
        if compute is None:
            return None
        tk, kind, groups = compute
        if kind == "binop":
            out_conn, a_conn, op, b_conn = groups
            in_conns = (a_conn, b_conn)
            if op not in _SVE_OP_INTRINSIC:
                return None
        else:
            out_conn, a_conn = groups
            in_conns = (a_conn, )
            op = None

        def _trace_in_array(conn):
            """Walk in-edges of the tasklet on ``conn`` back to the
            outer source AccessNode via memlet_path."""
            for ie in state.in_edges(tk):
                if ie.dst_conn != conn or ie.data is None or ie.data.data is None:
                    continue
                for hop in reversed(state.memlet_path(ie)):
                    if isinstance(hop.src, dace.nodes.AccessNode):
                        return hop.src.data
            return None

        def _trace_out_array(conn):
            """Walk out-edges through intermediate access nodes /
            scope exits to the outer destination AccessNode."""
            for oe in state.out_edges(tk):
                if oe.src_conn != conn or oe.data is None or oe.data.data is None:
                    continue
                # The compute may write to an intermediate AccessNode
                # which is then read by an assign tasklet that writes to
                # the outer destination. Walk until the OUTER access
                # node (one outside the map scope).
                seen = set()
                cur_edge = oe
                while True:
                    nxt = None
                    for hop in state.memlet_path(cur_edge):
                        if isinstance(hop.dst, dace.nodes.AccessNode):
                            nxt = hop.dst
                            break
                    if nxt is None or nxt in seen:
                        break
                    seen.add(nxt)
                    # If this access node feeds another tasklet that's
                    # an assign, follow through.
                    next_edges = state.out_edges(nxt)
                    if not next_edges:
                        return nxt.data
                    forward_to_assign = None
                    for ne in next_edges:
                        if isinstance(ne.dst, dace.nodes.Tasklet):
                            forward_to_assign = ne.dst
                            break
                        if isinstance(ne.dst, dace.nodes.MapExit):
                            # Outer access node is downstream of the exit.
                            for path_hop in state.memlet_path(ne):
                                if isinstance(path_hop.dst, dace.nodes.AccessNode):
                                    return path_hop.dst.data
                    if forward_to_assign is not None:
                        # Continue from this assign tasklet's out edge.
                        out_edges_assign = state.out_edges(forward_to_assign)
                        if not out_edges_assign:
                            return nxt.data
                        cur_edge = out_edges_assign[0]
                        continue
                    return nxt.data
            return None

        in_arrays = [_trace_in_array(c) for c in in_conns]
        out_array = _trace_out_array(out_conn)
        if out_array is None or any(a is None for a in in_arrays):
            return None
        # Reject kernels where the map reads/writes arrays beyond what
        # the recognised single-binop accounts for (e.g. a 3-operand
        # triad ``d = a + b + c``: my single-binop recogniser would
        # match the first add and silently drop ``c``). Strict equality
        # of the input-array set with the recognised inputs.
        outer_input_arrays = {
            ie.src.data
            for ie in state.in_edges(map_entry) if isinstance(ie.src, dace.nodes.AccessNode)
        }
        outer_output_arrays = {
            oe.dst.data
            for oe in state.out_edges(state.exit_node(map_entry)) if isinstance(oe.dst, dace.nodes.AccessNode)
        }
        if set(in_arrays) != outer_input_arrays or {out_array} != outer_output_arrays:
            return None
        for nm in (*in_arrays, out_array):
            arr = sdfg.arrays.get(nm)
            if arr is None or arr.dtype != dace.float64 or len(arr.shape) != 1:
                return None
        return (op, in_arrays, out_array, dace.float64, str(ub + 1))

    @staticmethod
    def _emit_sve_while_chain(state: dace.SDFGState, map_entry: dace.nodes.MapEntry, info):
        """Replace ``map_entry``'s scope with a single CPP tasklet
        emitting the SVE runtime-VL while-loop body that evaluates the
        recognised chain (each step is one SVE intrinsic). Handles
        single binop (axpy), longer linear chains (triad, etc.), and
        the copy degenerate case.

        :param info: ``(steps, out_array, global_ub_str)`` from
            :meth:`_classify_chain_body`.
        """
        steps, out_array, global_ub = info
        sdfg = state.sdfg
        map_exit = state.exit_node(map_entry)
        # Distinct outer input arrays in chain order — each gets a
        # connector on the new tasklet.
        seen = set()
        in_arrays = []
        for _op, outer_ins in steps:
            for arr in outer_ins:
                if arr not in seen:
                    seen.add(arr)
                    in_arrays.append(arr)
        in_conns = [f"_in{k}" for k in range(len(in_arrays))]
        arr_to_conn = {arr: c for arr, c in zip(in_arrays, in_conns)}
        out_conn = "_out"
        # Emit SVE body. ``acc`` holds the running result; after the
        # first step it's reused across the chain.
        sve_lines = ["int i = 0;", f"while (i < (int)({global_ub})) {{",
                     f"    svbool_t pg = svwhilelt_b64(i, (int64_t)({global_ub}));"]
        # Pre-load every outer input lane once. For long chains this is
        # the simplest correct emission (the compiler can hoist loads
        # if it likes); a more aggressive version would only load as
        # needed but micro-optimisation is out of scope here.
        for k, arr in enumerate(in_arrays):
            sve_lines.append(f"    svfloat64_t v{k} = svld1_f64(pg, {in_conns[k]} + i);")
        # Walk the steps.
        scalar_compute_parts = []
        prev = None
        for op, outer_ins in steps:
            if op == "copy":
                # Pure copy: only step in the chain.
                k = in_arrays.index(outer_ins[0])
                sve_lines.append(f"    svfloat64_t acc = v{k};")
                prev = "acc"
                scalar_compute_parts.append(f"{in_conns[k]}[i]")
                continue
            intrinsic = _SVE_OP_INTRINSIC[op]
            if prev is None:
                # First binop: two outer inputs.
                k0 = in_arrays.index(outer_ins[0])
                k1 = in_arrays.index(outer_ins[1])
                sve_lines.append(f"    svfloat64_t acc = {intrinsic}(pg, v{k0}, v{k1});")
                scalar_compute_parts.append(f"({in_conns[k0]}[i] {op} {in_conns[k1]}[i])")
                prev = "acc"
            else:
                # Subsequent: one new outer + the running acc.
                k = in_arrays.index(outer_ins[0])
                sve_lines.append(f"    acc = {intrinsic}(pg, acc, v{k});")
                scalar_compute_parts[-1] = f"({scalar_compute_parts[-1]} {op} {in_conns[k]}[i])"
        sve_lines.append(f"    svst1_f64(pg, {out_conn} + i, acc);")
        sve_lines.append("    i += svcntd();")
        sve_lines.append("}")
        sve_body = "\n".join("    " + ln for ln in sve_lines)
        scalar_rhs = scalar_compute_parts[0] if scalar_compute_parts else "0.0"
        body = ("\n#if defined(__ARM_FEATURE_SVE)\n"
                "{\n"
                f"{sve_body}\n"
                "}\n"
                "#else\n"
                f"    for (int i = 0; i < (int)({global_ub}); ++i) {out_conn}[i] = {scalar_rhs};\n"
                "#endif\n")
        guard = "#if defined(__ARM_FEATURE_SVE)\n#include <arm_sve.h>\n#endif\n"
        if guard not in sdfg.global_code.get("frame", dace.properties.CodeBlock("")).as_string:
            sdfg.append_global_code(guard, "frame")
        # Add the replacement tasklet and wire it to the outer access nodes.
        tk = state.add_tasklet("sve_vl", set(in_conns), {out_conn}, body, language=dace.dtypes.Language.CPP)
        outer_inputs = {ie.src.data for ie in state.in_edges(map_entry) if isinstance(ie.src, dace.nodes.AccessNode)}
        outer_outputs = {oe.dst.data for oe in state.out_edges(map_exit) if isinstance(oe.dst, dace.nodes.AccessNode)}
        out_access = {n.data: n for n in state.data_nodes() if n.data in outer_inputs | outer_outputs}
        for arr_name in in_arrays:
            src = out_access.get(arr_name) or state.add_access(arr_name)
            out_access[arr_name] = src
            shape0 = sdfg.arrays[arr_name].shape[0]
            state.add_edge(src, None, tk, arr_to_conn[arr_name], dace.Memlet(f"{arr_name}[0:{shape0}]"))
        dst = out_access.get(out_array) or state.add_access(out_array)
        shape0 = sdfg.arrays[out_array].shape[0]
        state.add_edge(tk, out_conn, dst, None, dace.Memlet(f"{out_array}[0:{shape0}]"))
        # Drop everything that used to be inside the map scope.
        for n in list(state.all_nodes_between(map_entry, map_exit)):
            if n not in (map_entry, map_exit) and n is not tk:
                state.remove_node(n)
        state.remove_node(map_entry)
        state.remove_node(map_exit)
        sdfg.validate()

    @staticmethod
    def _emit_sve_while_tasklet(state: dace.SDFGState, map_entry: dace.nodes.MapEntry, info):
        """Replace ``map_entry``'s scope with a single CPP tasklet whose
        body is the SVE runtime-VL while-loop (with scalar fallback)
        for the recognised op."""
        op, in_arrays, out_array, _dtype, global_ub = info
        sdfg = state.sdfg
        map_exit = state.exit_node(map_entry)
        # Collect outer access nodes (source/sink) that fed/consumed the
        # map. The replacement tasklet takes the SAME outer access nodes
        # via its in/out memlets (full-array slices).
        outer_inputs = {ie.src.data for ie in state.in_edges(map_entry) if isinstance(ie.src, dace.nodes.AccessNode)}
        outer_outputs = {oe.dst.data for oe in state.out_edges(map_exit) if isinstance(oe.dst, dace.nodes.AccessNode)}
        # Build the SVE while-loop body. For each input we emit a
        # svld1_f64 of the input array, for the op we emit the matching
        # _SVE_OP_INTRINSIC, and we svst1_f64 the result. Scalar
        # fallback for non-SVE hosts so x86 runs.
        n_in = len(in_arrays)
        in_conns = [f"_in{k}" for k in range(n_in)]
        out_conn = "_out"
        ld_lines = [f"svfloat64_t v{k} = svld1_f64(pg, {in_conns[k]} + i);" for k in range(n_in)]
        if op is None:
            # Copy.
            compute_line = "svfloat64_t vc = v0;"
            scalar_compute = f"{out_conn}[i] = {in_conns[0]}[i];"
        else:
            intrinsic = _SVE_OP_INTRINSIC[op]
            compute_line = f"svfloat64_t vc = {intrinsic}(pg, v0, v1);"
            scalar_compute = f"{out_conn}[i] = {in_conns[0]}[i] {op} {in_conns[1]}[i];"
        body = ("\n#if defined(__ARM_FEATURE_SVE)\n"
                "{\n"
                "    int i = 0;\n"
                f"    while (i < (int)({global_ub})) {{\n"
                f"        svbool_t pg = svwhilelt_b64(i, (int64_t)({global_ub}));\n"
                f"        " + "\n        ".join(ld_lines) + "\n"
                f"        {compute_line}\n"
                f"        svst1_f64(pg, {out_conn} + i, vc);\n"
                "        i += svcntd();\n"
                "    }\n"
                "}\n"
                "#else\n"
                f"    for (int i = 0; i < (int)({global_ub}); ++i) {scalar_compute}\n"
                "#endif\n")
        # Ensure the global SVE header inclusion happens exactly once
        # per SDFG (idempotent — set on the SDFG global_code dict).
        guard = "#if defined(__ARM_FEATURE_SVE)\n#include <arm_sve.h>\n#endif\n"
        if guard not in sdfg.global_code.get("frame", dace.properties.CodeBlock("")).as_string:
            sdfg.append_global_code(guard, "frame")
        # Build the replacement tasklet.
        tk = state.add_tasklet("sve_vl",
                               set(in_conns),
                               {out_conn},
                               body,
                               language=dace.dtypes.Language.CPP)
        # Outer access nodes: reuse the ones already in the state if
        # present, else add new ones bound to the map's source/dst.
        out_access = {n.data: n for n in state.data_nodes() if n.data in outer_inputs | outer_outputs}
        # Wire inputs: outer source access -> tasklet (full slice).
        for k, arr_name in enumerate(in_arrays):
            src = out_access.get(arr_name) or state.add_access(arr_name)
            out_access[arr_name] = src
            shape0 = sdfg.arrays[arr_name].shape[0]
            state.add_edge(src, None, tk, in_conns[k], dace.Memlet(f"{arr_name}[0:{shape0}]"))
        # Wire output.
        dst = out_access.get(out_array) or state.add_access(out_array)
        shape0 = sdfg.arrays[out_array].shape[0]
        state.add_edge(tk, out_conn, dst, None, dace.Memlet(f"{out_array}[0:{shape0}]"))
        # Remove the old map scope (entry, body, exit).
        for n in list(state.all_nodes_between(map_entry, map_exit)):
            if n not in (map_entry, map_exit) and n is not tk:
                # Body access nodes & tasklets internal to the old map.
                state.remove_node(n)
        state.remove_node(map_entry)
        state.remove_node(map_exit)
        sdfg.validate()

    @staticmethod
    def _classify_spmv_reduction_body(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
        """Recognise the SpMV-shape body: a single NSDFG (the dace
        frontend's reduction wrapping) containing a multiply tasklet on
        a gather (``b[idx[i]]``) + an augassign accumulator into a
        scalar output. Returns ``(a_arr, b_arr, idx_arr, out_scalar,
        global_ub_str)`` or ``None`` if not the SpMV shape.

        Pre-condition (must run before): WCRToAugAssign — converts the
        ``+=``-style WCR edge into an explicit augassign tasklet inside
        the NSDFG body.
        """
        if len(map_entry.map.params) != 1:
            return None
        lb, ub, step = map_entry.map.range[-1]
        if (step != 1) and (str(step) != "1"):
            return None
        sdfg = state.sdfg
        map_exit = state.exit_node(map_entry)
        body_nodes = [
            n for n in state.all_nodes_between(map_entry, map_exit)
            if not isinstance(n, (dace.nodes.MapEntry, dace.nodes.MapExit))
        ]
        # Body must be exactly one NSDFG (no other nodes).
        nsdfgs = [n for n in body_nodes if isinstance(n, dace.nodes.NestedSDFG)]
        if len(nsdfgs) != 1 or len(body_nodes) != 1:
            return None
        nsdfg_node = nsdfgs[0]
        inner = nsdfg_node.sdfg
        # Inner must have exactly one state with the multiply + augassign
        # + assign tasklets. (The dace frontend's reduction wrapping
        # produces a single state with 3 tasklets after WCRToAugAssign:
        # _Mult_, assign_, augassign.)
        inner_states = list(inner.all_states())
        compute_states = [st for st in inner_states if any(isinstance(n, dace.nodes.Tasklet) for n in st.nodes())]
        if len(compute_states) != 1:
            return None
        cst = compute_states[0]
        tasklets = [n for n in cst.nodes() if isinstance(n, dace.nodes.Tasklet)]
        mult = next((t for t in tasklets if t.code.as_string.strip().rstrip(";").strip() == "__out = (__in1 * __in2)"),
                    None)
        augassign = next(
            (t for t in tasklets if "augassign" in t.label and "+" in t.code.as_string), None)
        if mult is None or augassign is None:
            return None
        # Identify which outer arrays the NSDFG's input connectors map
        # to (traverse the outer in_edges).
        conn_to_outer = {}
        for ie in state.in_edges(nsdfg_node):
            if ie.dst_conn and ie.data is not None and ie.data.data is not None:
                conn_to_outer[ie.dst_conn] = ie.data.data
        # Inner inputs by NSDFG connector name: each maps to either ``a``
        # (loaded scalar at index i), ``b`` (gathered via ``idx``), or
        # ``idx`` (the integer index source).
        # Find the gather: inside cst, an access node whose data is an
        # NSDFG inner-array whose subset references a *symbol* (not the
        # literal map param). That symbol is the gather index = loaded
        # from idx.
        import re
        sym_re = re.compile(r"__sym_(\w+)")
        gather_arr_conn = None
        gather_index_conn = None
        for e in cst.edges():
            if e.data is None or e.data.data is None or e.data.subset is None:
                continue
            sub_str = str(e.data.subset)
            m = sym_re.search(sub_str)
            if m:
                # The accessed inner-array is the gather source; the
                # symbol whose name is __sym_<connector> maps to idx.
                # Map back to NSDFG outer connector via the inner array name.
                arr_name = e.data.data
                # Find which NSDFG input connector this inner array came from.
                # The inner array name typically matches the connector name.
                gather_arr_conn = arr_name
                gather_index_conn = m.group(1)
                break
        if gather_arr_conn is None or gather_index_conn is None:
            return None
        # b = arr behind the gather; idx = arr behind the index symbol's connector.
        b_outer = conn_to_outer.get(gather_arr_conn)
        idx_outer = conn_to_outer.get(gather_index_conn)
        # a = the third input (the scalar load at position i, NOT b and NOT idx).
        a_outer = None
        for c, outer_arr in conn_to_outer.items():
            if c != gather_arr_conn and c != gather_index_conn:
                a_outer = outer_arr
                break
        if a_outer is None or b_outer is None or idx_outer is None:
            return None
        # Output: the NSDFG's single outgoing edge to the outer scalar.
        out_scalar = None
        for oe in state.out_edges(map_exit):
            if isinstance(oe.dst, dace.nodes.AccessNode):
                out_scalar = oe.dst.data
                break
        if out_scalar is None:
            return None
        # Dtype + shape sanity.
        for nm in (a_outer, b_outer):
            arr = sdfg.arrays.get(nm)
            if arr is None or arr.dtype != dace.float64 or len(arr.shape) != 1:
                return None
        idx_arr = sdfg.arrays.get(idx_outer)
        if idx_arr is None or idx_arr.dtype != dace.int64 or len(idx_arr.shape) != 1:
            return None
        out_arr = sdfg.arrays.get(out_scalar)
        if out_arr is None or out_arr.dtype != dace.float64:
            return None
        return (a_outer, b_outer, idx_outer, out_scalar, str(ub + 1))

    @staticmethod
    def _emit_sve_spmv_reduction(state: dace.SDFGState, map_entry: dace.nodes.MapEntry, info):
        """Replace ``map_entry`` with a CPP tasklet doing the SpMV
        reduction via SVE intrinsics: svmla_f64_m accumulator over
        svwhilelt-predicated svld1_gather_s64index_f64 + svld1_f64,
        finalised by svaddv_f64. Scalar fallback for non-SVE hosts.
        """
        a_arr, b_arr, idx_arr, out_scalar, global_ub = info
        sdfg = state.sdfg
        map_exit = state.exit_node(map_entry)
        body = ("\n#if defined(__ARM_FEATURE_SVE)\n"
                "{\n"
                "    svfloat64_t acc = svdup_n_f64(0.0);\n"
                "    int i = 0;\n"
                f"    while (i < (int)({global_ub})) {{\n"
                f"        svbool_t pg = svwhilelt_b64(i, (int64_t)({global_ub}));\n"
                "        svint64_t vidx = svld1_s64(pg, _idx + i);\n"
                "        svfloat64_t va = svld1_f64(pg, _a + i);\n"
                "        svfloat64_t vbg = svld1_gather_s64index_f64(pg, _b, vidx);\n"
                "        acc = svmla_f64_m(pg, acc, va, vbg);\n"
                "        i += svcntd();\n"
                "    }\n"
                "    _out = svaddv_f64(svptrue_b64(), acc);\n"
                "}\n"
                "#else\n"
                "    {\n"
                "        double s = 0.0;\n"
                f"        for (int i = 0; i < (int)({global_ub}); ++i) s += _a[i] * _b[_idx[i]];\n"
                "        _out = s;\n"
                "    }\n"
                "#endif\n")
        guard = "#if defined(__ARM_FEATURE_SVE)\n#include <arm_sve.h>\n#endif\n"
        if guard not in sdfg.global_code.get("frame", dace.properties.CodeBlock("")).as_string:
            sdfg.append_global_code(guard, "frame")
        tk = state.add_tasklet("sve_spmv", {"_a", "_b", "_idx"}, {"_out"}, body, language=dace.dtypes.Language.CPP)
        # Wire to existing outer access nodes.
        outer_inputs = {ie.src.data for ie in state.in_edges(map_entry) if isinstance(ie.src, dace.nodes.AccessNode)}
        outer_outputs = {oe.dst.data for oe in state.out_edges(map_exit) if isinstance(oe.dst, dace.nodes.AccessNode)}
        out_access = {n.data: n for n in state.data_nodes() if n.data in outer_inputs | outer_outputs}
        for arr_name, conn in ((a_arr, "_a"), (b_arr, "_b"), (idx_arr, "_idx")):
            src = out_access.get(arr_name) or state.add_access(arr_name)
            out_access[arr_name] = src
            shape0 = sdfg.arrays[arr_name].shape[0]
            state.add_edge(src, None, tk, conn, dace.Memlet(f"{arr_name}[0:{shape0}]"))
        dst = out_access.get(out_scalar) or state.add_access(out_scalar)
        out_subset = "0" if len(sdfg.arrays[out_scalar].shape) == 0 else f"{out_scalar}[0:1]"
        state.add_edge(tk, "_out", dst, None, dace.Memlet(out_subset))
        for n in list(state.all_nodes_between(map_entry, map_exit)):
            if n not in (map_entry, map_exit) and n is not tk:
                state.remove_node(n)
        state.remove_node(map_entry)
        state.remove_node(map_exit)
        sdfg.validate()

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Replace every recognised innermost element-wise float64 map
        with the SVE runtime-VL while-loop tasklet.

        Dispatches per-map on body shape: SpMV-reduction (NSDFG with
        multiply + augassign over a gather) vs linear tasklet chain
        (axpy / triad / copy). The two shapes are recognised
        independently — no chaining across patterns.

        :param sdfg: SDFG to transform in place.
        :raises NotImplementedError: if any innermost map cannot be
            recognised as either pattern (no silent fallback).
        """
        # Convert any WCR (``+=`` produces it) into explicit sequential
        # augassign tasklets BEFORE the recogniser runs — so reductions
        # appear as ordinary augassign tasklets the SpMV recogniser
        # walks. Per user directive: "we should run WCRToAugAssign
        # beginning vectorization".
        from dace.transformation.dataflow.wcr_conversion import WCRToAugAssign
        sdfg.apply_transformations_repeated(WCRToAugAssign)
        targets = [(n, g) for n, g in list(sdfg.all_nodes_recursive())
                   if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState) and is_innermost_map(g, n)]
        applied = 0
        for n, g in targets:
            # Try the SpMV-reduction shape first (NSDFG body); fall
            # through to the flat tasklet chain shape (axpy/triad/copy)
            # — each pattern is recognised standalone, not chained.
            spmv_info = self._classify_spmv_reduction_body(g, n)
            if spmv_info is not None:
                self._emit_sve_spmv_reduction(g, n, spmv_info)
                applied += 1
                continue
            chain_info = self._classify_chain_body(g, n)
            if chain_info is not None:
                self._emit_sve_while_chain(g, n, chain_info)
                applied += 1
                continue
            raise NotImplementedError(
                f"sve_style='variable' recogniser supports: (1) SpMV-shape reduction "
                f"(NSDFG body with augassign over gather, after WCRToAugAssign), or "
                f"(2) linear chain of single-binop float64 1D element-wise tasklets "
                f"(axpy, triad, longer chains, or copy). Map {n.label!r} in state "
                f"{g.label!r} did not match either.")
        return applied or None
