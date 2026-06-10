# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tile MMA (matrix-multiply-accumulate) library node.

Per user direction 2026-06-10: K-dim register-tile MMA mirroring cuTile's
``ct.mma(a, b, c)`` primitive with GEMM-style ``alpha`` / ``beta`` scalar
prefactors (matching :class:`~dace.libraries.blas.nodes.gemm.Gemm` convention).

Tile shapes:

* ``_a``: ``(M, K_inner)`` -- left operand tile.
* ``_b``: ``(K_inner, N)`` -- right operand tile.
* ``_cin``: ``(M, N)`` -- accumulator input (only when ``beta != 0``).
* ``_c``: ``(M, N)`` -- result output.

The caller wires ``_cin`` and ``_c`` to the same AccessNode for in-place
accumulation -- mirrors DaCe's Tasklet convention (Tasklet codegen
disallows same-name in/out connectors, so the lib node carries distinct
names that the expansion preserves).

Compile-time properties:

* ``alpha`` -- scalar prefactor for ``A @ B`` (default 1).
* ``beta`` -- scalar prefactor for ``C`` (default 1 -- accumulate; set
  to 0 to overwrite, matching the ``ct.mma`` no-op-when-accumulator-empty
  convention).
* ``widths`` -- ``[M, K_inner, N]`` in source order. cuTile requires powers
  of 2 on every dim; the pure expansion accepts arbitrary positive ints
  (validated at expansion time).

The pure expansion emits a 3-fold nested CPP loop:

```cpp
for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
        T acc = T(0);
        for (size_t k = 0; k < K_inner; ++k) {
            acc += _a[i * K_inner + k] * _b[k * N + j];
        }
        _c[i * N + j] = T(alpha) * acc + T(beta) * _c[i * N + j];
    }
}
```

The cuTile expansion will lower to ``cuda.tile.mma`` once the cuTile
backend lands (stub raises ``NotImplementedError`` mirroring the existing
``ExpandTileBinopCutile`` placeholder).
"""
from typing import Optional, Tuple

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandTileMMAPure(ExpandTransformation):
    """Pure CPP expansion: 3-fold nested loop over (M, N, K_inner)."""

    environments = []

    @staticmethod
    def expansion(node: "TileMMA", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        node.validate(parent_sdfg, parent_state)
        M, K_inner, N = list(node.widths)
        # Pick the accumulator dtype from the ``_c`` output edge descriptor; the
        # input ``_a`` / ``_b`` dtypes must match (enforced in ``validate``).
        out_dtype = parent_sdfg.arrays[next(e for e in parent_state.out_edges(node)
                                            if e.src_conn == "_c").data.data].dtype.ctype
        alpha = node.alpha
        beta = node.beta
        lines = [
            f"for (std::size_t i = 0; i < {M}; ++i) {{",
            f"    for (std::size_t j = 0; j < {N}; ++j) {{",
            f"        {out_dtype} acc = {out_dtype}(0);",
            f"        for (std::size_t k = 0; k < {K_inner}; ++k) {{",
            f"            acc += _a[i * {K_inner} + k] * _b[k * {N} + j];",
            "        }",
        ]
        # Specialise the accumulator update at compile time -- avoid the
        # unnecessary ``* 1`` / ``+ 0`` arithmetic the compiler would otherwise
        # have to fold (and which obscures the generated code).
        if alpha == 1 and beta == 0:
            update = f"_c[i * {N} + j] = acc;"
        elif alpha == 1 and beta == 1:
            update = f"_c[i * {N} + j] = acc + _cin[i * {N} + j];"
        elif beta == 0:
            update = f"_c[i * {N} + j] = {out_dtype}({alpha}) * acc;"
        elif alpha == 1:
            update = f"_c[i * {N} + j] = acc + {out_dtype}({beta}) * _cin[i * {N} + j];"
        else:
            update = f"_c[i * {N} + j] = {out_dtype}({alpha}) * acc + {out_dtype}({beta}) * _cin[i * {N} + j];"
        lines += [f"        {update}", "    }", "}"]
        code = "\n".join(lines)

        # Inputs: ``_a``, ``_b`` always; ``_cin`` only when ``beta`` is non-zero
        # (we read the accumulator in that case). Output: always ``_c``.
        inputs = {"_a", "_b"} | ({"_cin"} if node.beta != 0 else set())
        return nodes.Tasklet(
            label=f"{node.label}_pure",
            inputs={c: None
                    for c in inputs},
            outputs={"_c": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


@library.expansion
class ExpandTileMMACutile(ExpandTransformation):
    """cuTile expansion: lowers to ``cuda.tile.mma(a, b, c)`` (stubbed)."""

    environments = []

    @staticmethod
    def expansion(node: "TileMMA", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        raise NotImplementedError(
            "ExpandTileMMACutile: cuTile expansion pending -- lowering will emit ``ct.mma(_a, _b, _c)`` "
            "wrapped in the alpha/beta arithmetic (alpha=1, beta=1 collapses to the bare ``ct.mma`` call). "
            "Pin a ``pure`` expansion via ``sdfg.expand_library_nodes(implementation='pure')`` for now.")


@library.node
class TileMMA(nodes.LibraryNode):
    """K-dim register-tile MMA: ``c = alpha * (a @ b) + beta * c`` in-place.

    Mirrors cuTile's ``ct.mma`` primitive with GEMM-style ``alpha`` / ``beta``
    compile-time scalar prefactors. The ``_c`` tile is both input (read for
    accumulation) and output (written in place).
    """

    implementations = {
        "pure": ExpandTileMMAPure,
        "cutile": ExpandTileMMACutile,
    }
    default_implementation = "pure"

    target_isa = properties.Property(
        dtype=str,
        allow_none=False,
        default="SCALAR",
        desc="Target ISA hint stamped by the orchestrator; consumed only by ISA-specific "
        "expansions (today only ``pure`` and ``cutile`` exist, so this is informational).",
    )
    widths = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Tile dimensions ``[M, K_inner, N]`` in source order. cuTile requires every dim to be "
        "a power of 2.",
    )
    alpha = properties.Property(
        allow_none=False,
        default=1,
        desc="Scalar prefactor for ``A @ B`` (matches :class:`Gemm.alpha` convention).",
    )
    beta = properties.Property(
        allow_none=False,
        default=1,
        desc="Scalar prefactor for the accumulator ``C``. Default 1 accumulates "
        "(``c = a @ b + c``); ``0`` overwrites (``c = a @ b``).",
    )

    def __init__(self, name: str, widths: Tuple[int, int, int], alpha=1, beta=1, location: Optional[str] = None):
        """Construct a ``TileMMA`` node.

        :param name: Node label.
        :param widths: ``(M, K_inner, N)`` tile dimensions.
        :param alpha: Compile-time scalar prefactor for ``A @ B``.
        :param beta: Compile-time scalar prefactor for the accumulator; 0
            overwrites, non-zero accumulates.
        :param location: Optional DaCe node location override.
        """
        if len(widths) != 3:
            raise ValueError(f"TileMMA: widths must be a 3-tuple (M, K_inner, N); got {widths!r}")
        if any(w <= 0 for w in widths):
            raise ValueError(f"TileMMA: every dim must be positive; got widths={widths!r}")
        # ``_cin`` is read for accumulation when ``beta != 0``; ``_c`` is
        # always the output (caller wires both to the same AccessNode for
        # in-place accumulation).
        inputs = {"_a", "_b"} | ({"_cin"} if beta != 0 else set())
        super().__init__(name, location=location, inputs=inputs, outputs={"_c"})
        self.widths = list(widths)
        self.alpha = alpha
        self.beta = beta

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Verify connector descriptors match the declared ``(M, K_inner, N)``
        and the dtypes of ``_a``, ``_b``, ``_cin`` / ``_c`` are uniform."""
        in_e = {e.dst_conn: e for e in state.in_edges(self)}
        out_e = {e.src_conn: e for e in state.out_edges(self)}
        for required in ("_a", "_b"):
            if required not in in_e:
                raise ValueError(f"{self.label}: missing required input connector {required!r}")
        if "_c" not in out_e:
            raise ValueError(f"{self.label}: missing required output connector '_c'")
        if self.beta != 0 and "_cin" not in in_e:
            raise ValueError(f"{self.label}: beta={self.beta!r} != 0 requires '_cin' input connector")
        M, K_inner, N = self.widths
        a_desc = sdfg.arrays[in_e["_a"].data.data]
        b_desc = sdfg.arrays[in_e["_b"].data.data]
        c_desc = sdfg.arrays[out_e["_c"].data.data]
        if a_desc.dtype != b_desc.dtype or a_desc.dtype != c_desc.dtype:
            raise NotImplementedError(f"{self.label}: requires uniform dtype across _a, _b, _c; "
                                      f"got a={a_desc.dtype}, b={b_desc.dtype}, c={c_desc.dtype}")
        for desc, name, expected in ((a_desc, "_a", (M, K_inner)), (b_desc, "_b", (K_inner, N)), (c_desc, "_c", (M,
                                                                                                                 N))):
            shape = tuple(desc.shape) if hasattr(desc, "shape") else ()
            if len(shape) != 2 or tuple(int(s) for s in shape) != expected:
                raise ValueError(f"{self.label}: {name!r} descriptor shape {shape} != expected {expected}")
