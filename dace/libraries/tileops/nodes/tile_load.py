# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileLoad`` — copy a K-dim tile out of a global array.

The source memlet's subset already describes the tile region in the
parent array; the pure expansion does a strided element-wise copy with
``dim_strides`` carrying the per-dim stride into the source view.
"""
from typing import List, Optional, Tuple

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandTileLoadPure(ExpandTransformation):
    """K-fold nested-Map copy with optional per-lane mask gating."""

    environments = []

    @staticmethod
    def expansion(node: "TileLoad", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        """Build a nested SDFG copying the tile region of ``_src`` into
        ``_dst``, gated by ``_mask`` when present.

        :param node: The ``TileLoad`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: Nested SDFG that replaces the lib node in place.
        """
        src_arr, dst_arr, mask_arr, src_subset = node.validate(parent_sdfg, parent_state)
        widths = list(node.widths)
        dim_strides = list(node.dim_strides) if node.dim_strides else [1] * len(widths)

        sdfg = dace.SDFG(f"{node.label}_pure")
        sdfg.add_array("_src", src_arr.shape, src_arr.dtype, strides=src_arr.strides, transient=False)
        sdfg.add_array("_dst", widths, dst_arr.dtype, transient=False)
        if node.has_mask:
            sdfg.add_array("_mask", widths, dace.bool_, transient=False)

        state = sdfg.add_state(f"{node.label}_state")
        K = len(widths)
        map_params = [f"__l{k}" for k in range(K)]
        map_rng = {p: f"0:{w}" for p, w in zip(map_params, widths)}
        dst_access = ", ".join(map_params)

        if src_subset is not None and len(src_subset) == len(src_arr.shape):
            base_exprs = []
            tile_dim = 0
            for d, (b, e, s) in enumerate(src_subset.ranges):
                if b == e:
                    base_exprs.append(str(b))
                else:
                    base_exprs.append(f"({b}) + {dim_strides[tile_dim]} * {map_params[tile_dim]}")
                    tile_dim += 1
            src_access = ", ".join(base_exprs)
        else:
            src_access = ", ".join(f"{dim_strides[k]} * {map_params[k]}" for k in range(K))

        if node.has_mask:
            body = "__output = __src if __mask else 0"
            inputs = {
                "__src": dace.Memlet(f"_src[{src_access}]"),
                "__mask": dace.Memlet(f"_mask[{dst_access}]"),
            }
        else:
            body = "__output = __src"
            inputs = {"__src": dace.Memlet(f"_src[{src_access}]")}
        outputs = {"__output": dace.Memlet(f"_dst[{dst_access}]")}

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
class TileLoad(nodes.LibraryNode):
    """Load a K-dim tile out of a global array.

    ``_src`` carries the full memlet of the source array; the in-edge's
    subset selects the tile region. ``_dst`` is the tile transient
    (``widths``-shaped). ``dim_strides`` records the per-tile-dim stride
    coefficient applied to the source index, defaulting to all 1s
    (contiguous).
    """

    implementations = {"pure": ExpandTileLoadPure}
    default_implementation = "pure"

    widths = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-dim tile widths, innermost-last.",
    )
    dim_strides = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-tile-dim stride into the source view; all 1s ⇒ contiguous.",
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
                 dim_strides: Optional[Tuple[int, ...]] = None,
                 has_mask: bool = False,
                 location: Optional[str] = None):
        """Construct a ``TileLoad`` node.

        :param name: Node label.
        :param widths: Per-dim tile widths, innermost-last.
        :param dim_strides: Per-tile-dim stride coefficients; defaults
            to all 1s (contiguous).
        :param has_mask: When True, declare the ``_mask`` input.
        :param location: Optional DaCe node location override.
        :raises ValueError: If ``widths`` is empty / longer than 3 or
            if ``dim_strides`` length disagrees with ``widths``.
        """
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileLoad: widths must have length in {{1, 2, 3}}, got {widths!r}")
        if dim_strides is not None and len(dim_strides) != len(widths):
            raise ValueError(
                f"TileLoad: dim_strides length {len(dim_strides)} != widths length {len(widths)}"
            )
        inputs = {"_src"} | ({"_mask"} if has_mask else set())
        super().__init__(name, location=location, inputs=inputs, outputs={"_dst"})
        self.widths = list(widths)
        self.dim_strides = list(dim_strides) if dim_strides else [1] * len(widths)
        self.has_mask = has_mask

    def validate(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
    ) -> Tuple[dace.data.Data, dace.data.Data, Optional[dace.data.Data], Optional[dace.subsets.Range]]:
        """Look up data descriptors and the source-edge subset.

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :returns: ``(src_arr, dst_arr, mask_arr_or_None,
            src_subset_or_None)``.
        :raises ValueError: If a required connector is unconnected.
        """
        in_e = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn is not None}
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if "_src" not in in_e:
            raise ValueError(f"{self.label}: required input '_src' not connected")
        if "_dst" not in out_e:
            raise ValueError(f"{self.label}: required output '_dst' not connected")
        if self.has_mask and "_mask" not in in_e:
            raise ValueError(f"{self.label}: has_mask=True but '_mask' not connected")
        src_arr = sdfg.arrays[in_e["_src"].data.data]
        dst_arr = sdfg.arrays[out_e["_dst"].data.data]
        mask_arr = sdfg.arrays[in_e["_mask"].data.data] if self.has_mask else None
        src_subset = in_e["_src"].data.subset
        return src_arr, dst_arr, mask_arr, src_subset
