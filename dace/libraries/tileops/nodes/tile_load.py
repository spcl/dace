# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileLoad`` — copy a K-dim tile out of a global array.

The pure expansion emits a CPP tasklet whose body walks the K-fold
nested index space using the source array's strides (which DaCe
codegen passes via ``__<arr>_strides`` from the surrounding scope).
"""
from typing import Optional, Tuple
from functools import reduce
from operator import mul

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandTileLoadPure(ExpandTransformation):
    """Correctness-only CPP tasklet copying the tile region into ``_dst``."""

    environments = []

    @staticmethod
    def expansion(node: "TileLoad", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a CPP tasklet that copies the tile region into the
        destination tile, optionally gated by ``_mask``.

        Source offsets use the source array's per-dim strides (read
        from the connector descriptor at expansion time) scaled by an
        optional :attr:`dim_strides` coefficient (defaulting to 1).

        :param node: The ``TileLoad`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A CPP tasklet replacing the lib node in place.
        """
        from dace.symbolic import symstr
        widths = list(node.widths)
        K = len(widths)
        src_edge = next(e for e in parent_state.in_edges(node) if e.dst_conn == "_src")
        src_arr = parent_sdfg.arrays[src_edge.data.data]
        src_strides = [symstr(s) for s in src_arr.strides[-K:]]
        coeff = list(node.dim_strides) if node.dim_strides else [1] * K
        n = reduce(mul, widths, 1)
        rem_var = "__rem"
        decoders = []
        for k in reversed(range(K)):
            decoders.append(f"        const std::size_t __l{k} = {rem_var} % {widths[k]};")
            decoders.append(f"        {rem_var} /= {widths[k]};")
        decoders_block = "\n".join(decoders)
        src_offset = " + ".join(
            f"({coeff[k]} * ({src_strides[k]}) * __l{k})" for k in range(K)
        ) if K else "0"
        if node.has_mask:
            body_inner = (
                f"        _dst[__k] = _mask[__k] ? _src[{src_offset}] : static_cast<std::remove_reference_t<decltype(_dst[__k])>>(0);"
            )
        else:
            body_inner = f"        _dst[__k] = _src[{src_offset}];"
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
class ExpandTileLoadCute(ExpandTransformation):
    """``cuda.tile``-Python expansion of :class:`TileLoad`.

    Emits a Python tasklet whose body calls ``cuda.tile.load`` with the
    surrounding-scope tile indices and ``mask=`` (when masked). The
    tile shape is the lib node's ``widths``.
    """

    environments = []

    @staticmethod
    def expansion(node: "TileLoad", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a Python tasklet emitting ``cuda.tile.load``.

        :param node: The lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A Python-language tasklet with a ``cuda.tile.load``
            body.
        """
        shape_tuple = ", ".join(str(w) for w in node.widths)
        if node.has_mask:
            body = f"__output = cuda.tile.load(__src, shape=({shape_tuple},), mask=__mask, padding_value=0)"
        else:
            body = f"__output = cuda.tile.load(__src, shape=({shape_tuple},))"
        inputs = {"__src"} | ({"__mask"} if node.has_mask else set())
        return nodes.Tasklet(
            label=f"{node.label}_cute",
            inputs={c: None for c in inputs},
            outputs={"__output": None},
            code=body,
            language=dace.dtypes.Language.Python,
        )


@library.node
class TileLoad(nodes.LibraryNode):
    """Load a K-dim tile out of a global array.

    ``_src`` carries the full memlet of the source array; the in-edge's
    subset selects the tile region. ``_dst`` is the tile transient
    (``widths``-shaped). ``dim_strides`` records the per-tile-dim stride
    coefficient applied to the source view, defaulting to all 1s
    (contiguous).
    """

    implementations = {"pure": ExpandTileLoadPure, "cute": ExpandTileLoadCute}
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
