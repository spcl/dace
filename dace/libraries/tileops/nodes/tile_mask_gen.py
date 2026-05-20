# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileMaskGen`` — allocate a K-dim ``bool`` mask whose lanes encode
the ANY-dim-OOB conjunction
``(i_0 + l_0 < ub_0) && ... && (i_{K-1} + l_{K-1} < ub_{K-1})``.

The pure expansion emits a CPP tasklet that writes the K-fold mask
into the linear ``_o`` buffer; the outer iter-vars and upper-bound
expressions are resolved from the surrounding scope.
"""
from typing import Optional, Tuple
from functools import reduce
from operator import mul

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandTileMaskGenPure(ExpandTransformation):
    """CPP tasklet writing the ANY-OOB conjunction into ``_o``."""

    environments = []

    @staticmethod
    def expansion(node: "TileMaskGen", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a CPP tasklet that fills the mask tile lane by lane.

        :param node: The ``TileMaskGen`` lib node being expanded.
        :param parent_state: State that owns the lib node (unused).
        :param parent_sdfg: SDFG that owns ``parent_state`` (unused).
        :returns: A CPP tasklet replacing the lib node in place.
        """
        widths = list(node.widths)
        iter_vars = list(node.iter_vars)
        global_ubs = list(node.global_ubs)
        n = reduce(mul, widths, 1)
        K = len(widths)
        rem_var = "__rem"
        decoders = []
        for k in reversed(range(K)):
            decoders.append(f"        const std::size_t __l{k} = {rem_var} % {widths[k]};")
            decoders.append(f"        {rem_var} /= {widths[k]};")
        decoders_block = "\n".join(decoders)
        terms = [f"((({iv}) + __l{k}) < ({ub}))"
                 for k, (iv, ub) in enumerate(zip(iter_vars, global_ubs))]
        cond = " && ".join(terms) if terms else "true"
        code = (
            f"for (std::size_t __k = 0; __k < {n}; ++__k) {{\n"
            f"        std::size_t {rem_var} = __k;\n"
            f"{decoders_block}\n"
            f"        _o[__k] = {cond};\n"
            f"}}"
        )
        return nodes.Tasklet(
            label=f"{node.label}_pure",
            inputs=set(),
            outputs={"_o": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


@library.node
class TileMaskGen(nodes.LibraryNode):
    """Produce the K-dim iteration mask ``bool[widths]``.

    No inputs; one output ``_o``. Each dim has a corresponding
    ``iter_vars[k]`` (the surrounding-map iter-var name) and
    ``global_ubs[k]`` (the original exclusive upper-bound expression).
    The lane ``(l_0, ..., l_{K-1})`` is active iff every per-dim check
    ``iter_var_k + l_k < global_ub_k`` holds.
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
        :param iter_vars: Per-dim outer iter-var name.
        :param global_ubs: Per-dim exclusive upper-bound expressions.
        :param location: Optional DaCe node location override.
        :raises ValueError: On length mismatch or invalid K.
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

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Confirm ``_o`` is connected and the descriptor has ``bool``.

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :raises ValueError: If ``_o`` is not connected or has the
            wrong dtype.
        """
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if "_o" not in out_e:
            raise ValueError(f"{self.label}: required output '_o' not connected")
        mask_arr = sdfg.arrays[out_e["_o"].data.data]
        if mask_arr.dtype != dace.bool_:
            raise ValueError(f"{self.label}: _o must have dtype bool_, got {mask_arr.dtype}")
