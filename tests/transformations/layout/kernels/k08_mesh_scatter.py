# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k08 mesh edge-residual scatter -- the Pad / Permute / Shuffle witness (OP2 airfoil ``res_calc``).

An unstructured-mesh edge residual: each edge carries a flux from its two incident cells, scattered
back onto those cells::

    f[e] = (x[c0[e]] - x[c1[e]]) * 0.5     (per-edge flux)
    y[c0[e]] += f[e] ;  y[c1[e]] -= f[e]   (edge -> cell scatter)

The scatter has write conflicts (two edges touch the same cell), so this port uses the RACE-FREE
padded-gather form (the OP2/coloring alternative): precompute, per cell, the table of incident edges
with their signs, valence-padded to a uniform width ``V`` (Pad -- a cell of degree ``< V`` gets its
unused slots filled with edge 0 and sign 0, which add nothing), then recast the scatter as a per-cell
gather-and-reduce::

    y[c] = sum_v  f[tab[c, v]] * tsg[c, v]

Two maps: the flux ``f`` is computed over edges into a single-writer transient (indirect gathers
``x[e0[e]]`` / ``x[e1[e]]``); then ``y`` is reduced over cells, the inner valence axis ``V`` folded
by a WCR map (``y`` pre-zeroed by ``run_closure``), gathering ``f`` through the flat incident-edge
table.

Layout decision -- the physical layout of the per-cell padded gather table ``tsg`` (the sign / weight
column of the ``(edge, sign)`` slot table):

    AoS  : ``[NC, V]``  -- a cell's V slots are contiguous   (identity)
    SoA  : ``[V, NC]``  -- slot ``v`` across all cells is contiguous (Permute, add_permute_maps)
    padded: ``[NC, V+p]`` -- the valence axis grown by trailing pad (Pad primitive)

Both are transparent (the Permute wrap-around transpose / the Pad's never-accessed trailing cells
preserve the result), so every candidate reproduces the numpy oracle; the sweep only picks the
physical layout.

Attribution:
  * M. Giles, G. Mudalige, et al., OP2 airfoil ``res_calc`` (OP-DSL/OP2-Common); Giles et al.,
    "Designing OP2 for GPU architectures," JPDC 2013.
  * A. Sulyok, G. Balogh, I. Reguly, G. Mudalige, "Locality optimized unstructured mesh algorithms
    on GPUs," JPDC 2019 (cell/edge reordering; the atomics-vs-gather trade).
  * SC26 layout-algebra paper (Pad = valence padding; Permute = AoS/SoA of the gather table).

A Shuffle candidate renumbers the per-cell field ``x`` by a closed-form cell bijection ``sigma``
(cyclic shift / reflection -- a mesh reorder). ``x`` is consumed through the runtime indirection
``x[e0[e]]``; ``ShuffleElements`` renumbers the store (``x'[i] = x[sigma(i)]``) and composes the
inverse into that consumer (``x'[sigma^{-1}(e0[e])]``), so the reorder is transparent and ``x`` is
still bound as-is.

Deferred (honest scope): the incident-edge INDEX table is passed FLAT (``tabf[NC*V]``) rather than as
a 2-D ``[NC, V]`` array, because a 2-D array used as a frontend indirection INDEX mis-generates C++
after the nested-SDFG boundary widening the layout passes require (the index subscript lowers to the
uncompilable ``tab[std::make_tuple(c, v)]``). The Permute/Pad witnesses therefore act on the 2-D sign
table ``tsg``; the edge-index table stays a flat CSR-style gather index.
"""
import numpy
import dace

from dace.libraries.layout.shuffle import register_shuffle
from dace.transformation.layout.brute_force import permutation_candidates, shuffle_candidates
from dace.transformation.layout.pad_dimensions import PadDimensions
from dace.transformation.layout.prepare import prepare_for_layout

NC, NE, V = dace.symbol("NC"), dace.symbol("NE"), dace.symbol("V")


@dace.program
def mesh_edge_residual(x: dace.float64[NC], e0: dace.int64[NE], e1: dace.int64[NE], tabf: dace.int64[NC * V],
                       tsg: dace.float64[NC, V], y: dace.float64[NC]):
    """Per-edge flux into a single-writer transient, then a per-cell WCR gather-reduce over V."""
    f = dace.define_local([NE], dace.float64)
    for e in dace.map[0:NE] @ dace.ScheduleType.Sequential:
        f[e] = (x[e0[e]] - x[e1[e]]) * 0.5
    for c, v in dace.map[0:NC, 0:V] @ dace.ScheduleType.Sequential:
        y[c] += f[tabf[c * V + v]] * tsg[c, v]


def grid_edges(n):
    """An ``n x n`` structured quad mesh as an UNstructured edge list (cells = grid points; right +
    down neighbours), so cell/edge orderings are non-trivial. Returns ``(edges[NE, 2], NC)``."""
    idx = numpy.arange(n * n, dtype=numpy.int64).reshape(n, n)
    right = numpy.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1)
    down = numpy.stack([idx[:-1].ravel(), idx[1:].ravel()], 1)
    return numpy.concatenate([right, down]), n * n


def make_padded_gather(e0, e1, ncells):
    """Cell <- incident-edge table, valence-padded (the Pad primitive): turns the conflicting scatter
    into a per-cell gather (race-free by construction). Unused slots stay ``(edge 0, sign 0)`` -- they
    read a valid edge and contribute nothing. Returns ``(tab[NC, V], tsg[NC, V])``."""
    inc = numpy.concatenate([e0, e1])
    sgn = numpy.concatenate([numpy.ones(e0.size), -numpy.ones(e1.size)])
    eid = numpy.concatenate([numpy.arange(e0.size), numpy.arange(e1.size)])
    order = numpy.argsort(inc, kind="stable")
    inc, sgn, eid = inc[order], sgn[order], eid[order]
    deg = numpy.bincount(inc, minlength=ncells)
    v = int(deg.max())
    tab = numpy.zeros((ncells, v), dtype=numpy.int64)
    tsg = numpy.zeros((ncells, v))
    slot = numpy.arange(inc.size) - numpy.repeat(numpy.cumsum(deg) - deg, deg)
    tab[inc, slot] = eid
    tsg[inc, slot] = sgn
    return numpy.ascontiguousarray(tab), numpy.ascontiguousarray(tsg)


def oracle(x, e0, e1, tab, tsg):
    """The dense padded-gather residual (== the ``np.add.at`` scatter, addition commutes): the
    per-edge flux gathered per cell and signed-summed over the valence axis."""
    f = (x[e0] - x[e1]) * 0.5
    return {"y": (f[tab] * tsg).sum(axis=1)}


def make_inputs(n=8, seed=0):
    """Small ``n x n`` mesh (``NC = n*n`` cells). Returns the per-cell field ``x``, the edge endpoint
    tables ``e0`` / ``e1``, and the valence-padded gather table ``tab`` (edge ids) / ``tsg`` (signs)."""
    edges, ncells = grid_edges(n)
    rng = numpy.random.default_rng(seed)
    e0 = numpy.ascontiguousarray(edges[:, 0])
    e1 = numpy.ascontiguousarray(edges[:, 1])
    x = numpy.ascontiguousarray(rng.random(ncells))
    tab, tsg = make_padded_gather(e0, e1, ncells)
    return {"x": x, "e0": e0, "e1": e1, "tab": tab, "tsg": tsg}


def _prepared(apply):
    """Wrap a layout ``apply`` so it first normalizes the SDFG (``prepare_for_layout``): this kernel
    has an indirection nested SDFG + a WCR reduction, so the passes need the widened normal form."""

    def wrapped(sdfg):
        prepare_for_layout(sdfg, validate=False)
        apply(sdfg)

    return wrapped


#: Closed-form cell reorderings for the Shuffle candidate. Both are self-inverse-checked bijections
#: on ``[0, NC)`` valid for ANY ``NC``: a cyclic shift and a reflection (an involution). Registered at
#: import so the sympy functions / C lowerings exist before ``candidates`` builds the sweep.
register_shuffle("mesh_cyc", "(i + 1) % NC", "(i + NC - 1) % NC")
register_shuffle("mesh_rev", "NC - 1 - i", "NC - 1 - i")


def candidates():
    """The global layout candidates:

      * ``identity``       -- AoS ``[NC, V]`` sign table, natural cell order.
      * ``permute_tsg_10`` -- SoA ``[V, NC]`` sign table (Permute, transparent add_permute_maps).
      * ``pad_tsg_2`` / ``pad_tsg_4`` -- grow the valence axis to ``[NC, V+p]`` (the Pad primitive).
      * ``shuffle_x_mesh_cyc`` / ``shuffle_x_mesh_rev`` -- renumber the cell field ``x`` by a
        closed-form mesh reorder (the Shuffle primitive; the reorder is transparent).

    Every candidate is transparent, so all reproduce the oracle -- the sweep only picks the layout.
    """
    cands = {"identity": (lambda sdfg: None)}
    # Permute family (AoS vs SoA), reusing the brute-force helper; drop the identity permutation
    # ([0, 1]) since it duplicates ``identity``.
    for name, apply in permutation_candidates("tsg", 2):
        if name.endswith("_01"):
            continue
        cands[name] = _prepared(apply)
    # Pad family: grow the valence dimension by each trailing pad amount.
    for extra in (2, 4):

        def pad_apply(sdfg, extra=extra):
            PadDimensions(pad_map={"tsg": [0, extra]}).apply_pass(sdfg, {})

        cands[f"pad_tsg_{extra}"] = _prepared(pad_apply)
    # Shuffle family: renumber ``x`` by each registered mesh reorder, reusing the brute-force helper
    # (skip its ``noshuffle`` identity, which duplicates ``identity``).
    for name, apply in shuffle_candidates("x", 0, ["mesh_cyc", "mesh_rev"]):
        if name.startswith("noshuffle"):
            continue
        cands[name] = _prepared(apply)
    return cands


def run_closure(inputs, n=8):
    """``run`` lays the sign table ``tsg`` out into each candidate's descriptor shape (identity AoS,
    padded ``[NC, V+p]``, or -- for the transparent Permute -- the unchanged ``[NC, V]`` external
    shape), flattens the incident-edge table for the indirect gather, and binds a fresh zeroed ``y``.
    """
    nc = int(inputs["x"].shape[0])
    ne = int(inputs["e0"].shape[0])
    v = int(inputs["tab"].shape[1])
    tabf = numpy.ascontiguousarray(inputs["tab"].reshape(-1))

    def pack_tsg(shape):
        # shape is the (evaluated) descriptor of ``tsg`` after the candidate applied.
        if tuple(shape) == (nc, v):  # identity AoS, or transparent Permute (external shape kept)
            return inputs["tsg"].copy()
        if len(shape) == 2 and shape[0] == nc and shape[1] > v:  # Pad: [NC, V+p]
            out = numpy.zeros(shape, dtype=numpy.float64)
            out[:, :v] = inputs["tsg"]
            return out
        raise ValueError(f"unexpected tsg descriptor shape {shape}")

    def run(sdfg):
        shape = tuple(int(dace.symbolic.evaluate(s, {NC: nc, NE: ne, V: v})) for s in sdfg.arrays["tsg"].shape)
        y = numpy.zeros(nc)
        sdfg(x=inputs["x"].copy(), e0=inputs["e0"].copy(), e1=inputs["e1"].copy(), tabf=tabf.copy(),
             tsg=pack_tsg(shape), y=y, NC=nc, NE=ne, V=v)
        return {"y": y}

    return run
