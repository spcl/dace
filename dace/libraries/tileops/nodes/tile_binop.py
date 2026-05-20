# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileBinop`` element-wise binary op on K-dim register tiles.

The node consumes two same-shape tiles ``_a`` and ``_b`` and produces a
same-shape tile ``_c``. When ``has_mask=True``, the optional ``_mask``
tile gates the write per lane. The pure expansion lowers to a K-fold
nested Map with one scalar tasklet per lane — correctness-only, no
SIMD; downstream T6 (AVX-512) and T9 (cuTile) emit the perf paths.
"""
from typing import List, Optional, Tuple

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

_PY_OP_RHS = {
    "+": "__rhs1 + __rhs2",
    "-": "__rhs1 - __rhs2",
    "*": "__rhs1 * __rhs2",
    "/": "__rhs1 / __rhs2",
    "min": "min(__rhs1, __rhs2)",
    "max": "max(__rhs1, __rhs2)",
    "<": "__rhs1 < __rhs2",
    "<=": "__rhs1 <= __rhs2",
    ">": "__rhs1 > __rhs2",
    ">=": "__rhs1 >= __rhs2",
    "==": "__rhs1 == __rhs2",
    "!=": "__rhs1 != __rhs2",
    "&&": "__rhs1 and __rhs2",
    "||": "__rhs1 or __rhs2",
}


@library.expansion
class ExpandTileBinopPure(ExpandTransformation):
    """Correctness-only K-fold nested-Map lowering of ``TileBinop``."""

    environments = []

    @staticmethod
    def expansion(node: "TileBinop", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        """Build a nested SDFG that loops over every lane and applies
        ``op`` to the corresponding ``_a`` and ``_b`` lanes, gated by
        ``_mask`` when present.

        :param node: The ``TileBinop`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: Nested SDFG that replaces the lib node in place.
        """
        a_arr, b_arr, c_arr, mask_arr = node.validate(parent_sdfg, parent_state)
        widths = list(node.widths)

        sdfg = dace.SDFG(f"{node.label}_pure")
        sdfg.add_array("_a", widths, a_arr.dtype, transient=False)
        sdfg.add_array("_b", widths, b_arr.dtype, transient=False)
        sdfg.add_array("_c", widths, c_arr.dtype, transient=False)
        if node.has_mask:
            sdfg.add_array("_mask", widths, dace.bool_, transient=False)

        state = sdfg.add_state(f"{node.label}_state")
        K = len(widths)
        map_params = [f"__l{k}" for k in range(K)]
        map_rng = {p: f"0:{w}" for p, w in zip(map_params, widths)}
        access = ", ".join(map_params)

        rhs = _PY_OP_RHS[node.op]
        if node.has_mask:
            body = f"__output = ({rhs}) if __mask else 0"
            inputs = {
                "__rhs1": dace.Memlet(f"_a[{access}]"),
                "__rhs2": dace.Memlet(f"_b[{access}]"),
                "__mask": dace.Memlet(f"_mask[{access}]"),
            }
        else:
            body = f"__output = {rhs}"
            inputs = {
                "__rhs1": dace.Memlet(f"_a[{access}]"),
                "__rhs2": dace.Memlet(f"_b[{access}]"),
            }
        outputs = {"__output": dace.Memlet(f"_c[{access}]")}

        state.add_mapped_tasklet(
            f"{node.label}_tasklet",
            map_rng,
            inputs,
            body,
            outputs,
            external_edges=True,
        )
        return sdfg


@library.node
class TileBinop(nodes.LibraryNode):
    """Element-wise binary op on K-dim register tiles.

    Inputs ``_a`` and ``_b`` and output ``_c`` all have the same tile
    shape (``widths``). ``has_mask=True`` enables the optional ``_mask``
    input (same shape, ``bool`` dtype) that gates the write per lane.

    :cvar implementations: Per-target expansions; ``"pure"`` is the
        K-fold nested-Map correctness fallback. AVX-512 and cuTile
        expansions are added in T6 / T9.
    :cvar default_implementation: ``"pure"`` until T6 lands.
    """

    implementations = {"pure": ExpandTileBinopPure}
    default_implementation = "pure"

    op = properties.Property(
        dtype=str,
        allow_none=False,
        default="+",
        desc="Binary op (one of: + - * / min max < <= > >= == != && ||).",
    )
    widths = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )
    has_mask = properties.Property(
        dtype=bool,
        allow_none=False,
        default=False,
        desc="When True, the ``_mask`` input connector is required.",
    )

    def __init__(self,
                 name: str,
                 widths: Tuple[int, ...],
                 op: str = "+",
                 has_mask: bool = False,
                 location: Optional[str] = None):
        """Construct a ``TileBinop`` node.

        :param name: Node label.
        :param widths: Per-dim tile widths, innermost-last.
        :param op: One of the keys of ``_PY_OP_RHS``.
        :param has_mask: When True, declare the ``_mask`` input
            connector. The emitter sets this when the node lives
            inside the masked-remainder body.
        :param location: Optional DaCe node location override.
        :raises ValueError: If ``op`` is unknown or ``widths`` is
            empty / longer than 3.
        """
        if op not in _PY_OP_RHS:
            raise ValueError(f"TileBinop: unknown op {op!r}; allowed: {sorted(_PY_OP_RHS)}")
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileBinop: widths must have length in {{1, 2, 3}}, got {widths!r}")
        inputs = {"_a", "_b"} | ({"_mask"} if has_mask else set())
        super().__init__(name, location=location, inputs=inputs, outputs={"_c"})
        self.widths = list(widths)
        self.op = op
        self.has_mask = has_mask

    def validate(self,
                 sdfg: dace.SDFG,
                 state: dace.SDFGState) -> Tuple[dace.data.Data, dace.data.Data, dace.data.Data, Optional[dace.data.Data]]:
        """Look up the data descriptors for every connector via the
        parent state's edges and check the homogeneous-dtype invariant.

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :returns: ``(a_arr, b_arr, c_arr, mask_arr_or_None)``.
        :raises ValueError: If a required connector is unconnected.
        :raises NotImplementedError: If ``_a``, ``_b`` and ``_c``
            dtypes are not all equal (E2 edge-case lock).
        """
        in_e = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn is not None}
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        for needed in ("_a", "_b"):
            if needed not in in_e:
                raise ValueError(f"{self.label}: required input {needed!r} not connected")
        if "_c" not in out_e:
            raise ValueError(f"{self.label}: required output '_c' not connected")
        if self.has_mask and "_mask" not in in_e:
            raise ValueError(f"{self.label}: has_mask=True but '_mask' not connected")
        a_arr = sdfg.arrays[in_e["_a"].data.data]
        b_arr = sdfg.arrays[in_e["_b"].data.data]
        c_arr = sdfg.arrays[out_e["_c"].data.data]
        if a_arr.dtype != b_arr.dtype or b_arr.dtype != c_arr.dtype:
            raise NotImplementedError(
                f"{self.label}: TileBinop requires uniform dtype across _a, _b, _c "
                f"(got {a_arr.dtype}, {b_arr.dtype}, {c_arr.dtype}); cast via separate tasklet first."
            )
        mask_arr = sdfg.arrays[in_e["_mask"].data.data] if self.has_mask else None
        return a_arr, b_arr, c_arr, mask_arr
