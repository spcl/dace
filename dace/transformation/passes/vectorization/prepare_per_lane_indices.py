# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Materialise per-lane gather indices into integer tile transients.

TILIFICATION_TRANSFORMATION_DESIGN.md section 3.8: for each gather
memlet inside a body NSDFG, this pass derives the lane-dependency set,
mints an integer transient whose shape is the Cartesian product of
widths over the dependent tile dims, fills it with a constant-
assignment tasklet, and wires it into the corresponding ``TileLoad`` /
``TileStore`` ``_idx_<d>`` connector.

This replaces the legacy 1D pipeline's ``_laneid_<i>`` symbol fan-out
machinery (``utils/lane_expansion.py``). The modern output is a tile
the lib node broadcasts at expansion time; no per-lane symbol survives
into the post-emit SDFG (see ``ClearPerLaneIndexSymbols`` post-audit,
design section 10.6).

Status (incremental landing):

* Step 1 (this commit): module-level helper
  :func:`materialise_per_lane_index_tile` for the single-dim
  dependency case (the bulk of cloudsc / ICON kernels). Pass scaffold
  with a no-op stub that documents the contract. Unit tests for the
  helper.
* Step 2 (follow-up): multi-dim dependency case (2-D / N-D index
  tiles).
* Step 3 (follow-up): pass walks every tile-tagged Map's body NSDFG,
  classifies gather memlets, calls the helper, rewires to TileLoad /
  TileStore.
"""
from typing import Any, Dict, Optional

import dace
from dace import dtypes
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation


def materialise_per_lane_index_tile(state: SDFGState,
                                    name_hint: str,
                                    gather_expr: str,
                                    tile_iter_vars,
                                    tile_widths,
                                    idx_dtype: dtypes.typeclass = dace.int64) -> str:
    """Materialise a per-lane gather index tile.

    Creates an integer transient of shape ``tuple(tile_widths)`` and
    emits a CPP tasklet at the current ``state`` that fills it with the
    per-lane evaluation of ``gather_expr``. The shape matches the
    Cartesian product of widths over the tile dims the expression
    depends on (design section 9.2 lane-dependency rule):

    | ``tile_iter_vars``    | shape              | per-lane body                       |
    |-----------------------|--------------------|-------------------------------------|
    | ``("i",)``            | ``(W_i,)``         | substitute ``i -> __l0`` per lane   |
    | ``("i", "j")``        | ``(W_i, W_j)``     | substitute ``i -> __l0, j -> __l1`` |
    | ``("i", "j", "k")``   | ``(W_i, W_j, W_k)``| ditto with ``__l2``                 |

    Accepts a single ``str`` + ``int`` (the K=1 short form) OR an
    iterable of ``str`` + ``int`` (the multi-dim form). The flat offset
    into the materialised tile is row-major over ``tile_widths`` so
    consumers can decode by stride.

    :param state: State to emit the materialisation tasklet into.
    :param name_hint: Hint for the transient name; uniquified via
        ``find_new_name=True``.
    :param gather_expr: The gather expression, with the
        ``tile_iter_vars`` as free variables. Tasklet body substitutes
        each ``var -> __l<idx>`` per lane.
    :param tile_iter_vars: Name(s) of the tile iter-vars the expression
        depends on. ``str`` for K=1; tuple / list for multi-dim.
    :param tile_widths: Tile width(s); shape matches ``tile_iter_vars``.
    :param idx_dtype: Descriptor dtype; ``int32`` or ``int64`` per
        design section 10.4.
    :returns: The materialised transient's name (AccessNode added to
        ``state``).
    """
    sdfg = state.sdfg
    # Normalise to tuples so the K=1 and K>=2 paths share one body.
    if isinstance(tile_iter_vars, str):
        iter_vars = (tile_iter_vars, )
        widths = (int(tile_widths), )
    else:
        iter_vars = tuple(tile_iter_vars)
        widths = tuple(int(w) for w in tile_widths)
    if len(iter_vars) != len(widths):
        raise ValueError(f"materialise_per_lane_index_tile: tile_iter_vars (len={len(iter_vars)}) "
                         f"and tile_widths (len={len(widths)}) must align")
    arr_name, _ = sdfg.add_array(name_hint,
                                 shape=widths,
                                 dtype=idx_dtype,
                                 storage=dtypes.StorageType.Register,
                                 transient=True,
                                 find_new_name=True)
    # Substitute each iter_var with its lane variable in the body.
    body_expr = gather_expr
    for d, var in enumerate(iter_vars):
        body_expr = body_expr.replace(var, f"(__l{d})")
    # Build the K-fold nested loop. The flat offset is row-major:
    # __l0 * (W_1 * W_2 * ...) + __l1 * (W_2 * ...) + ... + __l_{K-1}.
    K = len(widths)
    parts = []
    for i in range(K):
        inner = 1
        for q in range(i + 1, K):
            inner *= widths[q]
        parts.append(f"__l{i}" if inner == 1 else f"(__l{i} * {inner})")
    flat = " + ".join(parts) if parts else "0"
    code_lines = []
    for d in range(K):
        code_lines.append(f"{'    ' * d}for (std::size_t __l{d} = 0; __l{d} < {widths[d]}; ++__l{d}) {{")
    code_lines.append(f"{'    ' * K}_out[{flat}] = ({idx_dtype.ctype})({body_expr});")
    for d in reversed(range(K)):
        code_lines.append(f"{'    ' * d}}}")
    tasklet = state.add_tasklet(
        name=f"materialise_{arr_name}",
        inputs=set(),
        outputs={"_out"},
        code="\n".join(code_lines),
        language=dtypes.Language.CPP,
    )
    out_an = state.add_access(arr_name)
    out_subset = ", ".join(f"0:{w}" for w in widths)
    state.add_edge(tasklet, "_out", out_an, None, Memlet(f"{arr_name}[{out_subset}]"))
    return arr_name


@transformation.explicit_cf_compatible
class PreparePerLaneIndices(ppl.Pass):
    """Stub for the per-lane index materialisation pass (design section 3.8).

    Walks every tile-tagged Map's body NSDFG and, for each gather memlet,
    derives ``deps(d)`` (which tile iter-vars the gather expression touches),
    materialises an index tile via :func:`materialise_per_lane_index_tile`
    (or the multi-dim variant landed in step 2), and rewires the memlet
    through a ``TileLoad`` / ``TileStore`` ``_idx_<d>`` connector.

    This step-1 stub implements the **helper** and the pass scaffold; the
    full walk + classifier-driven rewrite lands in step 3 once G7 (the
    inside-body staging pass) has reduced every non-transient access to
    a canonical AN -> AN shape the classifier can read cleanly.
    """

    CATEGORY: str = "Vectorization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Stub -- the full walk lands in step 3.

        :returns: ``None`` (no rewrites performed by the stub).
        """
        return None
