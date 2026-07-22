# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""icon_zekinh -- the ICON DOUBLE-indirect (cell-from-edges) bilinear-interpolation witness.

ICON's kinetic-energy-at-cells reconstruction ``z_ekinh`` bilinearly interpolates the horizontal
kinetic energy ``z_kin_hor_e`` (defined on edges) onto cells. Every cell ``(jb, jc)`` gathers its
three incident edges through a pair of connectivity tables and combines them with the barycentric
bilinear weights ``e_bln``::

    z_ekinh[jb, jk, jc] = sum_{m=0,1,2} e_bln[jb, m, jc]
                          * z_kin_hor_e[edge_blk[jb, jc, m], jk, edge_idx[jb, jc, m]]

This is the semi-structured *double*-indirect sibling of e6 (a single edge->cell gather). ICON stores
edge data blocked as ``(nblock, nlev, nproma)``, so addressing one edge needs BOTH a block index
``edge_blk`` and an in-block index ``edge_idx`` -- two data-dependent gathers into the SAME field, on
dimensions 0 and 2, while the vertical level ``jk`` (dimension 1) is a plain affine axis. The three
terms of the ``m`` sum are written out explicitly (a fully unrolled elementwise reduction, no nested
python loop -- pure per-``(jb, jk, jc)`` dataflow).

Each ``z_kin_hor_e[edge_blk[...], jk, edge_idx[...]]`` lowers through ``add_indirection_subgraph``: an
Indirection tasklet reads the whole field via a dynamic memlet with the two connectivity entries fed
in as index memlets. The tables ``edge_idx`` / ``edge_blk`` are int64 inputs.

Two families of global layout decision are exposed, both transparent (``add_permute_maps`` wraps the
input and the permute recurses into the indirection's nested SDFG, so every candidate reproduces the
oracle -- the sweep only picks the physical layout):

  * the GATHERED field ``z_kin_hor_e`` -- ``[NB, NLEV, NPROMA]`` orderings (ICON's blocked
    ``(nblock, nlev, nproma)`` Fortran default vs level-contiguous / vertical-first, etc.). The permute
    carries through the double indirection because the whole gather lives in a nested SDFG whose
    memlets are permuted with the field.
  * the CONNECTIVITY tables ``edge_idx`` / ``edge_blk`` -- ``[NB, NPROMA, 3]`` orderings (which of the
    block / in-block / stencil-arm axes is contiguous). These are read by affine subscripts, so every
    permutation carries like e6's ``cell_idx`` / ``vert_idx``.

Blocking the gathered field ``z_kin_hor_e`` is NOT transparent here and is deferred: ``SplitDimensions``
raises the field's rank while the Indirection tasklet's index expression stays lower-rank, so the
gather reads the wrong element (same as e6; see ``candidates``).

Source: G. Zaengl, D. Reinert, P. Ripodas, M. Baldauf, "The ICON modelling framework of DWD and
MPI-M," QJRMS 141, 2015 (edge/cell connectivity, e_bln barycentric interpolation); M. Giorgetta et
al., "The ICON-A model for direct QBO simulations on GPUs," GMD 2022 (blocked (nproma, nlev, nblock)
storage order); SC26 layout paper (indirect-stencil connectivity layout).
"""
import numpy
import dace

from dace.transformation.layout.brute_force import permutation_candidates

NB, NLEV, NPROMA = dace.symbol("NB"), dace.symbol("NLEV"), dace.symbol("NPROMA")


@dace.program
def zekinh(e_bln: dace.float64[NB, 3, NPROMA], edge_idx: dace.int64[NB, NPROMA, 3],
           edge_blk: dace.int64[NB, NPROMA, 3], z_kin_hor_e: dace.float64[NB, NLEV, NPROMA],
           z_ekinh: dace.float64[NB, NLEV, NPROMA]):
    """Reconstruct cell kinetic energy: for each cell ``(jb, jc)`` and level ``jk``, gather the three
    incident edge columns of ``z_kin_hor_e`` through ``edge_blk`` / ``edge_idx`` and combine them with
    the barycentric weights ``e_bln``. The three ``m`` terms are written out (no WCR, no nested loop)."""
    for jb, jk, jc in dace.map[0:NB, 0:NLEV, 0:NPROMA] @ dace.ScheduleType.Sequential:
        z_ekinh[jb, jk, jc] = (
            e_bln[jb, 0, jc] * z_kin_hor_e[edge_blk[jb, jc, 0], jk, edge_idx[jb, jc, 0]] +
            e_bln[jb, 1, jc] * z_kin_hor_e[edge_blk[jb, jc, 1], jk, edge_idx[jb, jc, 1]] +
            e_bln[jb, 2, jc] * z_kin_hor_e[edge_blk[jb, jc, 2], jk, edge_idx[jb, jc, 2]])


def oracle(e_bln, edge_idx, edge_blk, z_kin_hor_e):
    """Pure-numpy vectorization of the double-indirect interpolation: for each stencil arm ``m`` fancy-
    index the block/in-block gather of ``z_kin_hor_e`` (the level axis is a plain ``arange`` broadcast),
    weight by ``e_bln[:, m, :]`` broadcast over the levels, and accumulate."""
    nb, nlev, nproma = z_kin_hor_e.shape
    lev = numpy.arange(nlev)[None, :, None]  # (1, NLEV, 1) affine level axis
    out = numpy.zeros((nb, nlev, nproma))
    for m in range(3):
        blk = edge_blk[:, :, m][:, None, :]  # (NB, 1, NPROMA)
        idx = edge_idx[:, :, m][:, None, :]  # (NB, 1, NPROMA)
        gathered = z_kin_hor_e[blk, lev, idx]  # (NB, NLEV, NPROMA)
        out += e_bln[:, m, :][:, None, :] * gathered
    return {"z_ekinh": out}


def make_inputs(nb, nlev, nproma, seed=0):
    """Small synthetic ICON-shaped inputs: random f64 weight / edge fields plus random-but-valid int64
    connectivity (each of a cell's three arms points at an in-range block and in-block index)."""
    rng = numpy.random.default_rng(seed)
    return {
        "e_bln": rng.random((nb, 3, nproma)),
        "edge_idx": rng.integers(0, nproma, size=(nb, nproma, 3)).astype(numpy.int64),
        "edge_blk": rng.integers(0, nb, size=(nb, nproma, 3)).astype(numpy.int64),
        "z_kin_hor_e": rng.random((nb, nlev, nproma)),
    }


def candidates():
    """The global layout candidates, each a single-array dimension permutation (identity included):

      * ``z_kin_hor_e`` -- gathered-field orientation over ``[NB, NLEV, NPROMA]`` (blocked ICON default
        vs level-contiguous, etc.). The permute carries through the double indirection's nested SDFG.
      * ``edge_idx`` / ``edge_blk`` -- connectivity-table orientation over ``[NB, NPROMA, 3]``. Read by
        affine subscripts, so every permutation carries (like e6's ``cell_idx`` / ``vert_idx``).

    Every permute is transparent (``add_permute_maps`` wraps the input), so all candidates reproduce
    the oracle; the sweep only picks the fastest physical layout.

    Blocking ``z_kin_hor_e`` is deliberately omitted: ``SplitDimensions`` raises the field's rank while
    the Indirection tasklet's index expression stays lower-rank, so the gather would read the wrong
    element -- deferred, same as e6.
    """
    cands = {}
    for arr in ("z_kin_hor_e", "edge_idx", "edge_blk"):
        cands.update(permutation_candidates(arr, 3))
    return cands


def _pack(arr, desc_shape):
    """Lay a logical input out into the candidate descriptor shape. Under ``add_permute_maps`` the
    input descriptor keeps its logical shape (the transpose happens inside the SDFG), so this is a
    plain copy; a real transpose branch is kept for a descriptor whose axes are a pure reversal."""
    if tuple(arr.shape) == tuple(desc_shape):
        return arr.copy()
    if tuple(desc_shape) == tuple(reversed(arr.shape)):
        return arr.T.copy()
    return arr.reshape(desc_shape).copy()


def run_closure(inputs, nb, nlev, nproma):
    """A ``run(sdfg) -> outputs`` closure: physically packs every input into whatever shape the
    candidate's descriptor specifies (read from ``sdfg.arrays[name].shape``) and binds a fresh zeroed
    ``z_ekinh`` each call. The output is never permuted, so it stays ``[NB, NLEV, NPROMA]``."""
    syms = {NB: nb, NLEV: nlev, NPROMA: nproma}

    def run(sdfg):
        args = {}
        for name, arr in inputs.items():
            desc_shape = tuple(int(dace.symbolic.evaluate(s, syms)) for s in sdfg.arrays[name].shape)
            args[name] = _pack(arr, desc_shape)
        out = numpy.zeros((nb, nlev, nproma))
        sdfg(NB=nb, NLEV=nlev, NPROMA=nproma, z_ekinh=out, **args)
        return {"z_ekinh": out}

    return run
