# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileMaskGen``: the iteration mask of a tile."""

import dace
from dace import library, properties
from dace.sdfg import nodes

from dace.libraries.tileops.environments import (
    TileOpsAVX2,
    TileOpsAVX512,
    TileOpsCUDA,
    TileOpsNeon,
    TileOpsScalar,
    TileOpsSVE,
)
from dace.libraries.tileops.expansions import ExpandTileIsa, ExpandTilePure
from dace.libraries.tileops.isa import require_k1
from dace.libraries.tileops.lanes import nested_loops, tile_offset
from dace.libraries.tileops.validation import validate_mask_descriptor_lock
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.optionals import required


@library.expansion
class ExpandTileMaskGenPure(ExpandTilePure):
    pass


@library.expansion
class ExpandTileMaskGenScalar(ExpandTileIsa):
    environments = [TileOpsScalar]
    backend = "scalar"


@library.expansion
class ExpandTileMaskGenAVX512(ExpandTileIsa):
    environments = [TileOpsAVX512]
    backend = "avx512"


@library.expansion
class ExpandTileMaskGenAVX2(ExpandTileIsa):
    environments = [TileOpsAVX2]
    backend = "avx2"


@library.expansion
class ExpandTileMaskGenNeon(ExpandTileIsa):
    environments = [TileOpsNeon]
    backend = "neon"


@library.expansion
class ExpandTileMaskGenSVE(ExpandTileIsa):
    environments = [TileOpsSVE]
    backend = "sve"


@library.expansion
class ExpandTileMaskGenCUDA(ExpandTileIsa):
    environments = [TileOpsCUDA]
    backend = "cuda"


@library.node
class TileMaskGen(TileOp):
    """The ``bool[widths]`` mask ``_o`` of the lanes that are in range.

    Lane ``(l_0, ..., l_{K-1})`` is active iff ``iter_vars[k] + l_k < global_ubs[k]`` for every dim ``k``, where the
    iteration variables and the exclusive upper bounds are expressions of the symbols in scope. A ``guard_predicate``
    is one more conjunct.
    """

    implementations = {
        "pure": ExpandTileMaskGenPure,
        "scalar": ExpandTileMaskGenScalar,
        "avx512": ExpandTileMaskGenAVX512,
        "avx2": ExpandTileMaskGenAVX2,
        "neon": ExpandTileMaskGenNeon,
        "sve": ExpandTileMaskGenSVE,
        "cuda": ExpandTileMaskGenCUDA,
    }
    default_implementation = "pure"

    iter_vars = properties.ListProperty(
        element_type=str,
        default=[],
        desc="Per-dim name of the iteration variable of the enclosing map.",
    )
    global_ubs = properties.ListProperty(
        element_type=str,
        default=[],
        desc="Per-dim exclusive upper bound, an expression of the symbols in scope.",
    )
    guard_predicate = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="A per-lane predicate AND-ed into the mask, a C++ expression over the lane indices ``__l0, __l1, ...`` "
        "and the symbols in scope, such as ``((i) + __l0) < (mid)``. It carries a branch guard the if-conversion "
        "could not lower, so the guarded region runs under the mask instead of as control flow.",
    )

    def __init__(
        self,
        name: str,
        widths: tuple[int, ...],
        iter_vars: tuple[str, ...],
        global_ubs: tuple[str, ...],
        guard_predicate: str | None = None,
        location: str | None = None,
    ):
        if not 1 <= len(widths) <= 3:
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
        self.guard_predicate = guard_predicate

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        out_edges = {edge.src_conn: edge for edge in state.out_edges(self) if edge.src_conn is not None}
        if "_o" not in out_edges:
            raise ValueError(f"{self.label}: required output '_o' not connected")
        validate_mask_descriptor_lock(
            self.label, "_o", sdfg.arrays[required(out_edges["_o"].data.data)], tuple(self.widths)
        )

    def can_lower_to_isa(self, state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
        # The headers build the mask from the bounds alone and would drop the guard, running every lane.
        return not self.guard_predicate

    def pure_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        terms = [
            f"((({iter_var}) + __l{dim}) < ({bound}))"
            for dim, (iter_var, bound) in enumerate(zip(self.iter_vars, self.global_ubs, strict=True))
        ]
        if self.guard_predicate:
            terms.append(f"({self.guard_predicate})")
        condition = " && ".join(terms) if terms else "true"
        return nodes.Tasklet(
            label=f"{self.label}_pure",
            inputs={},
            outputs={"_o": None},
            code=nested_loops(self.widths, f"_o[{tile_offset(self.widths)}] = {condition};"),
            language=dace.dtypes.Language.CPP,
        )

    def isa_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG, backend: str) -> nodes.Tasklet:
        self.validate(sdfg, state)
        vlen = require_k1(self)
        code = f"dace::tileops::tile_mask_gen<int64_t, {vlen}>(_o, ({self.iter_vars[0]}), ({self.global_ubs[0]}));"
        return nodes.Tasklet(
            label=f"{self.label}_{backend}",
            inputs={},
            outputs={"_o": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )
