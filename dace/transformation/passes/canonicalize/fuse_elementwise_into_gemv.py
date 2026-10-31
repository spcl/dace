# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lower a matrix-vector ``MatMul`` whose matrix is an elementwise map's private output to a WCR map, so fusion
removes the matrix.

Shape (nbody ``ax = (dx * inv_r3) @ mass``)::

    map i, j: M[i, j] = f(...)          map i: y[i] = 0
    y = M @ v                     ->    map i, j: y[i] += M[i, j] * v[j]

A GEMV streams its matrix once, so the BLAS call buys nothing over the map, while the ``N x N`` matrix costs a
write and a read per element. As a WCR map it fuses with its producer and the matrix becomes a scalar.

Contract, per ``MatMul`` at the top level of a state, one operand 2-D and the other 1-D:

* the matrix is a transient ``Array`` written in that state only by one map exit and read only by the ``MatMul``,
  with no other access node anywhere in its SDFG;
* every operand memlet and the output memlet is a step-1 range.

GEMM (both operands matrices) is compute-bound and stays a library node.
"""

from typing import Any

from dace import SDFG, data, properties, subsets
from dace.libraries.blas.nodes.matmul import MatMul
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.utils import find_new_name


def _unit_step(subset: subsets.Subset | None) -> bool:
    return isinstance(subset, subsets.Range) and all(s == 1 for _, _, s in subset.ranges)


def _offset(subset: subsets.Range, params: list[str]) -> str:
    """``subset`` indexed by ``params`` along its non-unit dimensions, its begin elsewhere."""
    idx, it = [], iter(params)
    for (b, e, _), size in zip(subset.ranges, subset.size()):
        idx.append(f"{b} + {next(it)}" if size != 1 else str(b))
    return ", ".join(idx)


@properties.make_properties
class FuseElementwiseIntoGemv(ppl.Pass):
    """Lower matrix-vector products over an elementwise map's private matrix to an init map and a WCR map."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        count = 0
        for g in sdfg.all_sdfgs_recursive():
            candidates = [
                (state, n)
                for state in g.states()
                for n in state.nodes()
                if isinstance(n, MatMul) and state.entry_node(n) is None
            ]
            if not candidates:
                continue
            access_count: dict[str, int] = {}
            for state in g.states():
                for an in state.data_nodes():
                    access_count[an.data] = access_count.get(an.data, 0) + 1
            for state, node in candidates:
                count += self._lower(g, state, node, access_count)
        return count or None

    @staticmethod
    def _lower(sdfg: SDFG, state: SDFGState, node: MatMul, access_count: dict[str, int]) -> int:
        ins = {e.dst_conn: e for e in state.in_edges(node)}
        outs = state.out_edges(node)
        if set(ins) != {"_a", "_b"} or len(outs) != 1:
            return 0
        a, b, out = ins["_a"], ins["_b"], outs[0]
        if not all(_unit_step(e.data.subset) for e in (a, b, out)) or not isinstance(out.dst, nodes.AccessNode):
            return 0
        dims = [len([s for s in e.data.subset.size() if s != 1]) for e in (a, b)]
        if dims == [2, 1]:
            mat, vec, mat_params, vec_params, out_params = a, b, ["__gv_i", "__gv_j"], ["__gv_j"], ["__gv_i"]
        elif dims == [1, 2]:
            mat, vec, mat_params, vec_params, out_params = b, a, ["__gv_i", "__gv_j"], ["__gv_i"], ["__gv_j"]
        else:
            return 0
        src = mat.src
        desc = sdfg.arrays.get(mat.data.data)
        if (
            not isinstance(src, nodes.AccessNode)
            or not isinstance(desc, data.Array)
            or isinstance(desc, data.View)
            or not desc.transient
            or access_count.get(src.data, 0) != 1
            or state.out_degree(src) != 1
            or state.in_degree(src) != 1
            or not isinstance(state.in_edges(src)[0].src, nodes.MapExit)
        ):
            return 0
        sizes = [s for s in mat.data.subset.size() if s != 1]
        ranges = {p: f"0:{s}" for p, s in zip(mat_params, sizes)}
        out_sizes = [s for s in out.data.subset.size() if s != 1]
        if len(out_sizes) != 1:
            return 0
        out_name, out_subset = out.data.data, out.data.subset
        mat_name, mat_subset = mat.data.data, mat.data.subset
        vec_name, vec_subset = vec.data.data, vec.data.subset
        mat_node, vec_node, out_node = src, vec.src, out.dst
        if not isinstance(vec_node, nodes.AccessNode):
            return 0
        state.remove_node(node)
        # A connector may not take the name of a container, symbol or constant of the SDFG
        taken = set(sdfg.arrays) | set(sdfg.symbols) | set(sdfg.constants)
        m, v, o = (find_new_name(name, taken) for name in ("__gv_m", "__gv_v", "__gv_o"))
        # y = 0, then y[i] += M[i, j] * v[j]; the zeroed node orders the init before the accumulation
        zeroed = state.add_access(out_name)
        out_param = out_params[0]
        state.add_mapped_tasklet(
            f"{node.label}_init",
            {out_param: ranges[out_param]},
            {},
            f"{o} = 0",
            {o: Memlet(f"{out_name}[{_offset(out_subset, out_params)}]")},
            external_edges=True,
            output_nodes={out_name: zeroed},
        )
        _, entry, _ = state.add_mapped_tasklet(
            f"{node.label}_gemv",
            ranges,
            {
                m: Memlet(f"{mat_name}[{_offset(mat_subset, mat_params)}]"),
                v: Memlet(f"{vec_name}[{_offset(vec_subset, vec_params)}]"),
            },
            f"{o} = {m} * {v}",
            {o: Memlet(f"{out_name}[{_offset(out_subset, out_params)}]", wcr="lambda x, y: x + y")},
            external_edges=True,
            input_nodes={mat_name: mat_node, vec_name: vec_node},
            output_nodes={out_name: out_node},
        )
        state.add_nedge(zeroed, entry, Memlet())
        return 1

    def report(self, pass_retval: int) -> str:
        return f"Lowered {pass_retval} matrix-vector product(s) over a private elementwise matrix to WCR maps."
