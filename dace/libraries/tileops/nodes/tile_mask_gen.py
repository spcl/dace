# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileMaskGen`` — allocate a K-dim ``bool`` mask whose lanes encode the
ANY-dim-OOB conjunction ``(i_0 + l_0 < ub_0) && ... && (i_{K-1} + l_{K-1}
< ub_{K-1})``.

The mask lives in register storage and is consumed by the tile lib
nodes inside the masked-remainder body. Producer-only node — no inputs.
"""
from typing import List, Optional, Tuple

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandTileMaskGenPure(ExpandTransformation):
    """K-fold nested-Map emitter for the ANY-dim-OOB mask."""

    environments = []

    @staticmethod
    def expansion(node: "TileMaskGen", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        """Build a nested SDFG whose body writes ``_o[l_*] = AND_k
        (iter_var_k + l_k < global_ub_k)`` for every lane combination.

        :param node: The ``TileMaskGen`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: Nested SDFG that replaces the lib node in place.
        """
        mask_arr, = node.validate(parent_sdfg, parent_state)
        widths = list(node.widths)
        iter_vars = list(node.iter_vars)
        global_ubs = list(node.global_ubs)

        sdfg = dace.SDFG(f"{node.label}_pure")
        sdfg.add_array("_o", widths, dace.bool_, transient=False)

        state = sdfg.add_state(f"{node.label}_state")
        K = len(widths)
        map_params = [f"__l{k}" for k in range(K)]
        map_rng = {p: f"0:{w}" for p, w in zip(map_params, widths)}
        access = ", ".join(map_params)

        cond_terms = [f"(({iv}) + {lp} < ({ub}))" for iv, lp, ub in zip(iter_vars, map_params, global_ubs)]
        body = "__output = " + " and ".join(cond_terms)

        state.add_mapped_tasklet(
            f"{node.label}_tasklet",
            map_rng,
            {},
            body,
            {"__output": dace.Memlet(f"_o[{access}]")},
            external_edges=True,
        )
        return sdfg


@library.node
class TileMaskGen(nodes.LibraryNode):
    """Produce the K-dim iteration mask ``bool[widths]``.

    No inputs; one output ``_o``. Each dim has a corresponding
    ``iter_vars[k]`` (the surrounding-map iter-var name) and
    ``global_ubs[k]`` (the original exclusive upper-bound expression).
    The lane ``(l_0, ..., l_{K-1})`` is active iff every per-dim
    half-open check ``iter_var_k + l_k < global_ub_k`` holds.
    """

    implementations = {"pure": ExpandTileMaskGenPure}
    default_implementation = "pure"

    widths = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-dim mask widths, innermost-last.",
    )
    iter_vars = properties.ListProperty(
        element_type=str,
        default=[],
        desc="Per-dim outer-map iter-var name; symbol is resolved in the surrounding scope.",
    )
    global_ubs = properties.ListProperty(
        element_type=str,
        default=[],
        desc="Per-dim exclusive upper-bound expressions; symbols resolved in the surrounding scope.",
    )

    def __init__(self,
                 name: str,
                 widths: Tuple[int, ...],
                 iter_vars: Tuple[str, ...],
                 global_ubs: Tuple[str, ...],
                 location: Optional[str] = None):
        """Construct a ``TileMaskGen`` node.

        :param name: Node label.
        :param widths: Per-dim mask widths, innermost-last.
        :param iter_vars: Per-dim outer iter-var name (length matches
            ``widths``).
        :param global_ubs: Per-dim exclusive upper-bound expressions
            (length matches ``widths``).
        :param location: Optional DaCe node location override.
        :raises ValueError: If the three lists have mismatched lengths,
            or ``widths`` is empty / longer than 3.
        """
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileMaskGen: widths length {len(widths)} not in {{1, 2, 3}}")
        if len(iter_vars) != len(widths) or len(global_ubs) != len(widths):
            raise ValueError(
                f"TileMaskGen: widths / iter_vars / global_ubs lengths must agree; "
                f"got {len(widths)}, {len(iter_vars)}, {len(global_ubs)}"
            )
        super().__init__(name, location=location, inputs=set(), outputs={"_o"})
        self.widths = list(widths)
        self.iter_vars = list(iter_vars)
        self.global_ubs = list(global_ubs)

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> Tuple[dace.data.Data]:
        """Confirm ``_o`` is connected and its descriptor has the
        expected ``bool`` dtype.

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :returns: ``(mask_arr,)`` (one-element tuple to mirror the
            other lib nodes' validate signature).
        :raises ValueError: If ``_o`` is not connected or has the
            wrong dtype.
        """
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if "_o" not in out_e:
            raise ValueError(f"{self.label}: required output '_o' not connected")
        mask_arr = sdfg.arrays[out_e["_o"].data.data]
        if mask_arr.dtype != dace.bool_:
            raise ValueError(f"{self.label}: _o must have dtype bool_, got {mask_arr.dtype}")
        return (mask_arr,)
