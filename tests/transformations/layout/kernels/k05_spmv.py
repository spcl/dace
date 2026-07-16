# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k05 SpMV under formats and orderings -- the Pad + Block + Permute witness (ELL sparse gather).

Sparse matrix-vector product ``y[i] = sum_j val[j] * x[col[j]]`` (CSR). The port uses the ELL
(dense padded) form of the same graph: every row is padded to the maximum vertex degree ``W`` and
stored as a dense ``[NR, W]`` gather, where padded lanes point at the row's own index with value 0
(so they contribute ``0 * x[i] = 0`` and the ELL product equals the CSR product exactly)::

    y[i] = sum_k eval_[i, k] * x[ecol[i, k]]          (ELL, k over the W padded columns)

The padding to ``W = maxdeg`` is the **Pad** primitive; row tiling of the ELL matrix is **Block**;
the row-major ELL (``eval_[NR, W]``) vs column-major SELL slab (``eval_.T[W, NR]``) is **Permute**.
The gather ``x[ecol[i, k]]`` is a genuine data-dependent indirection (an ``add_indirection_subgraph``
lowering), which is what makes the sparse layout decision real rather than a dense reshape.

The kernel is written as gather-then-reduce (two maps over the same ``[0:NR, 0:W]`` space)::

    g[i, k] = x[ecol[i, k]]                 # gather: the only indirect map
    y[i]   += eval_[i, k] * g[i, k]         # reduce: a plain WCR map (mvt idiom, y pre-zeroed)

Separating the gather from the reduce keeps ``eval_`` out of the indirection's nested SDFG, so the
Block and Pad passes (which recurse through descriptors) can reshape ``eval_`` -- while ``ecol`` and
``x`` remain inside the indirection subgraph and are reached by the transparent input-wrapping
Permute (``add_permute_maps``). The result is identical to the single-line SpMV; the two maps only
expose more of the ELL layout to the sweep.

Layout decisions the sweep explores: the storage orientation of the value/index matrices (row-major
ELL vs column-major SELL slab -- Permute), row tiling of the value matrix (Block), and padding the
ragged width up to a SIMD multiple (Pad). Every candidate is transparent, so all reproduce the
oracle; the sweep only picks the physical layout.

Source: M. Kreutzer, G. Hager, G. Wellein, H. Fehske, A. Bishop, "A Unified Sparse Matrix Data Format
for Efficient General SpMV on Modern Processors with Wide SIMD Units," SIAM J. Sci. Comput. 36(5),
2014 (SELL-C-sigma = Permute+Block+Pad+Permute); E. Cuthill, J. McKee, "Reducing the bandwidth of
sparse symmetric matrices," ACM 1969 (vertex ordering); SC26 layout-algebra paper SS IV (sparse
gather isolation: Pad width, Block rows, Permute ELL<->SELL).
"""
import numpy
import dace

from dace.transformation.layout.brute_force import block_candidates, permutation_candidates
from dace.transformation.layout.pad_dimensions import PadDimensions

NR = dace.symbol("NR")  # number of rows (matrix and output length)
NC = dace.symbol("NC")  # number of columns (length of the gathered vector x)
W = 4  # ELL width == max vertex degree of grid5 (an interior 5-point node has 4 neighbours)


@dace.program
def spmv_ell(ecol: dace.int64[NR, W], eval_: dace.float64[NR, W], x: dace.float64[NC], y: dace.float64[NR]):
    """ELL SpMV as gather-then-reduce: ``y[i] = sum_k eval_[i,k] * x[ecol[i,k]]`` (y pre-zeroed)."""
    g = numpy.empty((NR, W), dtype=numpy.float64)  # per-row gathered x values (transient, fully written)
    for i, k in dace.map[0:NR, 0:W]:
        g[i, k] = x[ecol[i, k]]  # the indirect gather (data-dependent read of x)
    for i, k in dace.map[0:NR, 0:W] @ dace.ScheduleType.Sequential:
        y[i] += eval_[i, k] * g[i, k]  # WCR reduction over the W columns


# --------------------------------------------------------------- graph -> ELL (numpy spec) --------- #
def grid5(side):
    """2D ``side x side`` 5-point Laplacian as a symmetric edge list (upper triangle), ``NR = side^2``."""
    idx = numpy.arange(side * side).reshape(side, side)
    right = numpy.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
    down = numpy.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)
    return numpy.concatenate([right, down]), side * side


def to_csr(edges, n):
    """Symmetrize the edge list into CSR arrays ``(indptr, col, val, row-of-nnz)``."""
    src = numpy.concatenate([edges[:, 0], edges[:, 1]])
    dst = numpy.concatenate([edges[:, 1], edges[:, 0]])
    order = numpy.lexsort((dst, src))
    src, dst = src[order], dst[order]
    indptr = numpy.zeros(n + 1, dtype=numpy.int64)
    numpy.add.at(indptr, src + 1, 1)
    indptr = numpy.cumsum(indptr)
    val = numpy.ones(dst.shape[0], dtype=numpy.float64)
    return indptr, numpy.ascontiguousarray(dst), val, numpy.ascontiguousarray(src)


def make_ell(indptr, col, val, n):
    """Pad CSR to a dense ``(n, maxdeg)`` ELL: padded columns point at the row's own index, value 0."""
    deg = numpy.diff(indptr)
    w = int(deg.max())
    ecol = numpy.repeat(numpy.arange(n, dtype=numpy.int64)[:, None], w, axis=1)
    eval_ = numpy.zeros((n, w), dtype=numpy.float64)
    live = numpy.arange(w)[None, :] < deg[:, None]
    ecol[live] = col
    eval_[live] = val
    return numpy.ascontiguousarray(ecol), numpy.ascontiguousarray(eval_), w


def spmv_csr(indptr, col, val, row, x, n):
    """Reference CSR SpMV, used only to cross-check the ELL oracle in the sweep driver."""
    return numpy.bincount(row, weights=val * x[col], minlength=n)


def oracle(ecol, eval_, x):
    """The ELL SpMV ``y = (eval_ * x[ecol]).sum(axis=1)`` -- equal to ``spmv_csr`` on the same graph."""
    return {"y": (eval_ * x[ecol]).sum(axis=1)}


def make_inputs(side=8, seed=0):
    """Build the grid5 graph, convert to CSR, pad to ELL, and draw a random ``x`` (int64 index array)."""
    edges, n = grid5(side)
    indptr, col, val, row = to_csr(edges, n)
    ecol, eval_, w = make_ell(indptr, col, val, n)
    if w != W:
        raise ValueError(f"k05_spmv: grid5(side={side}) has max degree {w}, expected W={W}")
    x = numpy.random.default_rng(seed).random(n)
    return {"ecol": ecol, "eval_": eval_, "x": x}


def candidates():
    """The global layout candidates for the ELL matrices:

      * ``permute_eval_*`` / ``permute_ecol_*`` -- row-major ELL ``[NR, W]`` vs column-major SELL slab
        ``[W, NR]`` (Permute of the value / index matrix; the input-wrapping form reaches even the
        indexed ``ecol`` inside the gather's indirection subgraph).
      * ``noblock_eval_`` / ``block_eval__d0_*`` -- row tiling of the value matrix (Block; the row
        axis ``NR`` divides the SIMD factors, the width axis ``W`` does not so it is not tiled).
      * ``pad_eval__w8`` -- grow the ragged width ``W`` to a SIMD multiple (Pad); the extra lanes are
        never read (the map iterates ``0:W``), so the pad is transparent.

    Every candidate is transparent, so all reproduce the oracle. Block/Pad are offered on ``eval_``
    only: ``ecol`` and ``x`` live inside the gather's indirection nested SDFG, which those descriptor-
    recursing passes do not rewrite (see the module docstring / deferred notes)."""
    cands = {}
    cands.update(permutation_candidates("eval_", 2))  # permute_eval__01 (id), permute_eval__10 (SELL)
    cands.update(permutation_candidates("ecol", 2))  # permute_ecol_01 (id), permute_ecol_10 (SELL)
    for name, apply in block_candidates("eval_", 2, factors=(8, 16, 32)):
        if "_d1_" in name:  # the width axis W=4 does not divide the SIMD factors -> tile the row axis only
            continue
        cands[name] = apply  # noblock_eval_, block_eval__d0_{8,16,32}

    def pad_eval_w8(sdfg):
        PadDimensions(pad_map={"eval_": [0, W]}).apply_pass(sdfg, {})  # W -> 2*W trailing pad lanes

    cands["pad_eval__w8"] = pad_eval_w8
    return cands


def pack_ell(arr, shape, nr, fill):
    """Lay the logical ``[NR, W]`` ELL matrix ``arr`` out into a candidate descriptor ``shape``:

      * ``[NR, W]``        -- identity, plain copy.
      * ``[NR, W + p]``    -- Pad: live columns first, ``p`` trailing ``fill`` lanes (never read).
      * ``[W, NR]``        -- Permute (column-major SELL slab): transpose.
      * ``[NR/T, W, T]``   -- Block (row axis tiled, tile grouped last): reshape ``[NR/T, T, W]`` then
                              swap the tile axis to the end (mirrors k02.pack_a).
    """
    if len(shape) == 2:
        if shape[0] == nr:  # row-major, possibly width-padded
            out = numpy.full(shape, fill, dtype=arr.dtype)
            out[:, :W] = arr
            return out
        return numpy.ascontiguousarray(arr.T)  # column-major SELL slab [W, NR]
    nb, ww, t = shape
    return numpy.ascontiguousarray(arr.reshape(nb, t, ww).transpose(0, 2, 1))


def run_closure(inputs, side):
    """A ``run(sdfg) -> {"y": ...}`` closure for the sweep: physically packs ``ecol`` and ``eval_``
    into whatever layout each candidate's descriptor specifies (Permute / Block / Pad), binds ``x``
    as-is and a fresh zeroed ``y``, and returns the natural-order result for the oracle comparison."""
    n = side * side

    def run(sdfg):
        ecol_shape = tuple(int(dace.symbolic.evaluate(s, {NR: n, NC: n})) for s in sdfg.arrays["ecol"].shape)
        eval_shape = tuple(int(dace.symbolic.evaluate(s, {NR: n, NC: n})) for s in sdfg.arrays["eval_"].shape)
        ecol_in = pack_ell(inputs["ecol"], ecol_shape, n, 0)  # pad lanes never read -> any in-range index
        eval_in = pack_ell(inputs["eval_"], eval_shape, n, 0.0)
        y = numpy.zeros(n)
        sdfg(ecol=ecol_in, eval_=eval_in, x=inputs["x"].copy(), y=y, NR=n, NC=n)
        return {"y": y}

    return run
