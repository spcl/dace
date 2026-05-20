# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileStore`` — write a K-dim tile back into a global array.

Symmetric to :class:`TileLoad`; the pure expansion emits a CPP tasklet
that walks the K-fold nested index space.
"""
from typing import Optional, Tuple
from functools import reduce
from operator import mul

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandTileStorePure(ExpandTransformation):
    """Correctness-only CPP tasklet copying ``_src`` into the tile region of ``_dst``."""

    environments = []

    @staticmethod
    def expansion(node: "TileStore", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a CPP tasklet that copies ``_src`` into the tile
        region of the destination, optionally gated by ``_mask``.

        Destination offsets use the destination array's per-dim strides
        (read from the connector descriptor at expansion time) scaled
        by an optional :attr:`dim_strides` coefficient (defaulting to 1).

        :param node: The ``TileStore`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A CPP tasklet replacing the lib node in place.
        """
        from dace.symbolic import symstr
        widths = list(node.widths)
        K = len(widths)
        dst_edge = next(e for e in parent_state.out_edges(node) if e.src_conn == "_dst")
        dst_arr = parent_sdfg.arrays[dst_edge.data.data]
        dst_strides = [symstr(s) for s in dst_arr.strides[-K:]]
        coeff = list(node.dim_strides) if node.dim_strides else [1] * K
        n = reduce(mul, widths, 1)
        rem_var = "__rem"
        decoders = []
        for k in reversed(range(K)):
            decoders.append(f"        const std::size_t __l{k} = {rem_var} % {widths[k]};")
            decoders.append(f"        {rem_var} /= {widths[k]};")
        decoders_block = "\n".join(decoders)
        dst_offset = " + ".join(
            f"({coeff[k]} * ({dst_strides[k]}) * __l{k})" for k in range(K)
        ) if K else "0"
        if node.has_mask:
            body_inner = f"        if (_mask[__k]) {{ _dst[{dst_offset}] = _src[__k]; }}"
        else:
            body_inner = f"        _dst[{dst_offset}] = _src[__k];"
        code = (
            f"for (std::size_t __k = 0; __k < {n}; ++__k) {{\n"
            f"        std::size_t {rem_var} = __k;\n"
            f"{decoders_block}\n"
            f"{body_inner}\n"
            f"}}"
        )
        inputs = {"_src"} | ({"_mask"} if node.has_mask else set())
        return nodes.Tasklet(
            label=f"{node.label}_pure",
            inputs={c: None for c in inputs},
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


@library.expansion
class ExpandTileStoreCute(ExpandTransformation):
    """``cuda.tile``-Python expansion of :class:`TileStore`.

    Emits a Python tasklet whose body calls ``cuda.tile.store`` with the
    surrounding-scope tile indices and ``mask=`` (when masked).
    """

    environments = []

    @staticmethod
    def expansion(node: "TileStore", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a Python tasklet emitting ``cuda.tile.store``.

        :param node: The lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A Python-language tasklet with a ``cuda.tile.store``
            body.
        """
        if node.has_mask:
            body = "cuda.tile.store(__output, tile=__src, mask=__mask)"
        else:
            body = "cuda.tile.store(__output, tile=__src)"
        inputs = {"__src"} | ({"__mask"} if node.has_mask else set())
        return nodes.Tasklet(
            label=f"{node.label}_cute",
            inputs={c: None for c in inputs},
            outputs={"__output": None},
            code=body,
            language=dace.dtypes.Language.Python,
        )


@library.node
class TileStore(nodes.LibraryNode):
    """Store a K-dim tile back into a global array.

    ``_src`` is the tile transient (``widths``-shaped); ``_dst`` carries
    the full memlet of the destination array with the out-edge's subset
    selecting the tile region. ``dim_strides`` records per-tile-dim
    strides into the destination view.
    """

    implementations = {"pure": ExpandTileStorePure, "cute": ExpandTileStoreCute}
    default_implementation = "pure"

    widths = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-dim tile widths, innermost-last.",
    )
    dim_strides = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-tile-dim stride into the destination view; all 1s ⇒ contiguous.",
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
        """Construct a ``TileStore`` node.

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
            raise ValueError(f"TileStore: widths must have length in {{1, 2, 3}}, got {widths!r}")
        if dim_strides is not None and len(dim_strides) != len(widths):
            raise ValueError(
                f"TileStore: dim_strides length {len(dim_strides)} != widths length {len(widths)}"
            )
        inputs = {"_src"} | ({"_mask"} if has_mask else set())
        super().__init__(name, location=location, inputs=inputs, outputs={"_dst"})
        self.widths = list(widths)
        self.dim_strides = list(dim_strides) if dim_strides else [1] * len(widths)
        self.has_mask = has_mask

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Check connectors.

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
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
