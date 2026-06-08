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
                                    tile_iter_var: str,
                                    tile_width: int,
                                    idx_dtype: dtypes.typeclass = dace.int64) -> str:
    """Materialise a 1-D per-lane gather index tile.

    Creates a transient of shape ``(tile_width,)``, ``int64`` (or
    ``idx_dtype``), and emits a constant-assignment tasklet at the
    current ``state`` that fills it with the per-lane evaluation of
    ``gather_expr``. The tile is suitable for wiring into a
    :class:`TileLoad` / :class:`TileStore` ``_idx_<d>`` connector.

    Per design section 9.2: this is the ``deps = (p,)`` case where the
    gather expression depends on exactly one tile iter-var. The flat
    offset into the index tile at lane ``l_p`` is just ``l_p``.

    :param state: State to emit the materialisation tasklet into.
    :param name_hint: Hint for the transient name; will be uniquified
        via ``find_new_name=True``.
    :param gather_expr: The gather expression, with ``tile_iter_var`` as
        a free variable. For example ``idx[i + 3]`` with
        ``tile_iter_var="i"``. The expression must NOT itself contain
        AccessNode subscripts the body would need to resolve -- the
        caller pre-materialises any such inner reads before calling
        this helper.
    :param tile_iter_var: The name of the tile iter-var the expression
        depends on (substituted with the lane index in the emitted
        tasklet body).
    :param tile_width: ``W_p`` -- the tile width for the dependent dim.
    :param idx_dtype: Descriptor dtype for the materialised tile.
        Default ``int64``; ``int32`` is also acceptable per design
        section 10.4.
    :returns: The name of the materialised transient (already added
        to ``state.sdfg.arrays``; an AccessNode for it is added to
        ``state``).
    """
    sdfg = state.sdfg
    # Mint the transient.
    arr_name, _ = sdfg.add_array(name_hint,
                                 shape=(tile_width, ),
                                 dtype=idx_dtype,
                                 storage=dtypes.StorageType.Register,
                                 transient=True,
                                 find_new_name=True)
    # Emit one CPP tasklet that materialises the lane vector.
    # The per-lane body substitutes ``tile_iter_var`` with the lane index ``__l0``.
    # Use a simple flat materialisation loop; no fancy intrinsics needed (the
    # caller will lower the load that consumes this tile via TileLoad's pure
    # expansion).
    body_expr = gather_expr.replace(tile_iter_var, "(__l0)")
    code_lines = [
        f"for (std::size_t __l0 = 0; __l0 < {tile_width}; ++__l0) {{",
        f"    _out[__l0] = ({idx_dtype.ctype})({body_expr});",
        "}",
    ]
    tasklet = state.add_tasklet(
        name=f"materialise_{arr_name}",
        inputs=set(),
        outputs={"_out"},
        code="\n".join(code_lines),
        language=dtypes.Language.CPP,
    )
    out_an = state.add_access(arr_name)
    state.add_edge(tasklet, "_out", out_an, None, Memlet(f"{arr_name}[0:{tile_width}]"))
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
