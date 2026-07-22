# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""icon_one_loop -- ICON velocity one-loop vertical block stencil (multi-output, vertical backref).

A single ICON velocity ``one_loop`` sweep over a 3D field ``[NB, NLEV, NPROMA]`` (block index ``jb``,
vertical level ``jk``, in-block cell ``je``) writes two tendencies per interior level. It is the
DIRECT structured sibling of k07: a fixed vertical ``jk-1`` backward reference (a level-column
dependence), but with TWO independent outputs instead of one::

    vn_ie[jb, jk, je] = vn[jb, jk, je] - vn[jb, jk - 1, je]
    zkh[jb, jk, je]   = vt[jb, jk, je] - wgtfac_e[jb, jk, je]

The sweep runs over ``jk in [1, NLEV)`` so level ``jk = 0`` is never written and stays zero (ICON's
top-half-level tendency is undefined). The first output differences the field ``vn`` against the level
below it (the semi-structured vertical stencil); the second is a pure elementwise level difference of
``vt`` and ``wgtfac_e``.

The layout decision is the dimension order of the fields -- one global Permute per field. In C-order
the last axis is unit-stride, so the identity ``[0,1,2]`` (NPROMA last) is the ICON
``(nblock, nlev, nproma)`` order where each contiguous run is one horizontal cell-row of a level
(HORIZONTAL-first for the in-block index); a permutation putting ``NLEV`` last makes each level column
contiguous (VERTICAL-first), which is what the ``jk``/``jk-1`` backward reference streams. ``vn`` is
the vertical-stride-sensitive field (it is the one read at both ``jk`` and ``jk-1``), so its
permutation is the primary lever, exactly as ``A`` is in k07; ``vt`` and ``wgtfac_e`` are also swept
because their level-difference read is layout-sensitive too. Every permute is transparent
(``add_permute_maps`` wraps the input and permutes each memlet subset), so every candidate reproduces
the oracle -- the sweep only picks the fastest physical layout.

Source: G. Zaengl, D. Reinert, P. Ripodas, M. Baldauf, "The ICON modelling framework of DWD and
MPI-M," QJRMS 141, 2015 (velocity tendencies, vertical half-level stencil); M. Giorgetta et al., "The
ICON-A model for direct QBO simulations on GPUs," GMD 2022 ((nproma, nlev, nblocks) unit-stride
order); SC26 layout paper SS IV-D.
"""
import numpy
import dace

from dace.transformation.layout.brute_force import permutation_candidates

NB, NLEV, NPROMA = dace.symbol("NB"), dace.symbol("NLEV"), dace.symbol("NPROMA")


@dace.program
def one_loop(vn: dace.float64[NB, NLEV, NPROMA], wgtfac_e: dace.float64[NB, NLEV, NPROMA],
             vt: dace.float64[NB, NLEV, NPROMA], vn_ie: dace.float64[NB, NLEV, NPROMA],
             zkh: dace.float64[NB, NLEV, NPROMA]):
    """One velocity one-loop sweep over interior levels ``jk in [1, NLEV)``: difference ``vn`` against
    the level below it into ``vn_ie`` and difference ``vt``/``wgtfac_e`` into ``zkh``. Pure
    elementwise per ``(jb, jk, je)`` (no reduction); ``jk = 0`` is never written and stays 0."""
    for jb, jk, je in dace.map[0:NB, 1:NLEV, 0:NPROMA] @ dace.ScheduleType.Sequential:
        vn_ie[jb, jk, je] = vn[jb, jk, je] - vn[jb, jk - 1, je]
        zkh[jb, jk, je] = vt[jb, jk, je] - wgtfac_e[jb, jk, je]


def oracle(vn, wgtfac_e, vt):
    """Pure-numpy reference: both outputs start at zero (level 0 never written), the interior levels
    are the vertical backward difference of ``vn`` and the level difference of ``vt``/``wgtfac_e``."""
    vn_ie = numpy.zeros_like(vn)
    zkh = numpy.zeros_like(vt)
    vn_ie[:, 1:, :] = vn[:, 1:, :] - vn[:, :-1, :]
    zkh[:, 1:, :] = vt[:, 1:, :] - wgtfac_e[:, 1:, :]
    return {"vn_ie": vn_ie, "zkh": zkh}


def make_inputs(nb=2, nlev=16, nproma=8, seed=0):
    """Small synthetic ICON-shaped fields: ``vn``, ``wgtfac_e``, ``vt`` random ``[NB, NLEV, NPROMA]``."""
    rng = numpy.random.default_rng(seed)
    return {
        "vn": rng.random((nb, nlev, nproma)),
        "wgtfac_e": rng.random((nb, nlev, nproma)),
        "vt": rng.random((nb, nlev, nproma)),
    }


def candidates():
    """The global layout candidates: every dimension permutation of each read field (identity
    included). ``vn`` is the vertical-stride-sensitive field read at ``jk`` and ``jk-1``; ``vt`` and
    ``wgtfac_e`` are the level-difference operands. Every permute is transparent (``add_permute_maps``
    wraps the input), so all candidates reproduce the oracle -- the sweep only picks the fastest."""
    cands = {}
    for arr in ("vn", "vt", "wgtfac_e"):
        cands.update(permutation_candidates(arr, 3))
    return cands


def _pack(arr, desc_shape):
    """Lay a logical input out into the candidate descriptor shape. Under ``add_permute_maps`` the
    input descriptor keeps its logical shape (the transpose happens inside the SDFG), so this is a
    plain copy; a real transpose branch is kept for a descriptor whose axes have been reordered."""
    if tuple(arr.shape) == tuple(desc_shape):
        return arr.copy()
    if sorted(desc_shape) == sorted(arr.shape):
        perm = [arr.shape.index(s) for s in desc_shape]
        return numpy.transpose(arr, perm).copy()
    return arr.reshape(desc_shape).copy()


def run_closure(inputs, nb=2, nlev=16, nproma=8):
    """A ``run(sdfg) -> outputs`` closure for the sweep: physically packs every input into whatever
    shape the candidate's descriptor specifies (read from ``sdfg.arrays[name].shape``) and binds fresh
    zeroed ``vn_ie`` / ``zkh`` each call. The outputs are never permuted, so they stay
    ``[NB, NLEV, NPROMA]`` (and their untouched level 0 stays 0)."""
    syms = {NB: nb, NLEV: nlev, NPROMA: nproma}

    def run(sdfg):
        args = {}
        for name, arr in inputs.items():
            desc_shape = tuple(int(dace.symbolic.evaluate(s, syms)) for s in sdfg.arrays[name].shape)
            args[name] = _pack(arr, desc_shape)
        vn_ie = numpy.zeros((nb, nlev, nproma))
        zkh = numpy.zeros((nb, nlev, nproma))
        sdfg(NB=nb, NLEV=nlev, NPROMA=nproma, vn_ie=vn_ie, zkh=zkh, **args)
        return {"vn_ie": vn_ie, "zkh": zkh}

    return run
