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

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Replace every recognised innermost element-wise float64 map
        with the SVE runtime-VL while-loop tasklet.

        :param sdfg: SDFG to transform in place.
        :raises NotImplementedError: if any innermost map cannot be
            recognised as a simple element-wise pattern (no silent
            fallback — the user opted into ``sve_style='variable'``
            knowing the recogniser is first-cut narrow).
        """
        targets = [(n, g) for n, g in list(sdfg.all_nodes_recursive())
                   if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState) and is_innermost_map(g, n)]
        applied = 0
        for n, g in targets:
            info = self._classify_simple_axpy_body(g, n)
            if info is None:
                raise NotImplementedError(
                    f"sve_style='variable' first-cut recogniser supports only simple float64 element-wise "
                    f"``out[i] = a[i] (op) b[i]`` or ``out[i] = a[i]`` 1D innermost maps. "
                    f"Map {n.label!r} in state {g.label!r} did not match.")
            self._emit_sve_while_tasklet(g, n, info)
            applied += 1
        return applied or None
