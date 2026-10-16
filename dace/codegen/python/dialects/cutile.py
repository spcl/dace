# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""cuTile Python: a kernel takes arrays and reads them by per-dim index tiles, a tile is a ``ct`` tile."""

from dace import dtypes
from dace.codegen.python.kernel import Dialect, Element

#: Imported by the generated module: cuTile and the kernel spellings of the SymPy helpers. cuTile has no fused
#: multiply-add, so ``tile_fma`` rounds twice where the other backends round once.
IMPORTS = """\
import cuda.tile as ct


@ct.function
def tile_int_floor(a, b):
    return ct.floordiv(a, b)


@ct.function
def tile_int_ceil(a, b):
    return -ct.floordiv(-a, b)


@ct.function
def tile_fma(a, b, c):
    return a * b + c
"""


class CuTile(Dialect):
    imports = IMPORTS
    decorator = "@ct.kernel"
    module = "ct"
    functions = {
        "int_floor": "tile_int_floor",
        "int_ceil": "tile_int_ceil",
        "fma": "tile_fma",
        "Min": "ct.minimum",
        "Max": "ct.maximum",
        "Abs": "ct.abs",
        "ipow": "ct.pow",
        "fmod": "ct.mod",
    }

    def program_id(self, axis: int) -> str:
        return f"ct.bid({axis})"

    def lane(self, widths: list[int], dim: int) -> str:
        lanes = f"ct.arange({widths[dim]}, dtype=ct.int32)"
        if len(widths) == 1:
            return lanes
        shape = tuple(width if k == dim else 1 for k, width in enumerate(widths))
        return f"ct.reshape({lanes}, {shape})"

    def load(self, base: str, at: Element, mask: str | None, dtype: dtypes.typeclass) -> str:
        masked = "" if mask is None else f", mask={mask}, padding_value=0"
        return f"ct.gather({base}, ({', '.join(at.indices)},){masked})"

    def store(self, base: str, at: Element, value: str, mask: str | None) -> str:
        masked = "" if mask is None else f", mask={mask}"
        return f"ct.scatter({base}, ({', '.join(at.indices)},), {value}{masked})"

    def cast_type(self, dtype: dtypes.typeclass) -> str:
        return f"ct.{ {dtypes.bool_: 'bool_'}.get(dtype, dtype.to_string()) }"

    def cast(self, value: str, dtype: dtypes.typeclass) -> str:
        return f"ct.astype({value}, {self.cast_type(dtype)})"

    def full(self, widths: list[int], value: str, dtype: dtypes.typeclass) -> str:
        return f"ct.full({tuple(widths)}, {value}, {self.cast_type(dtype)})"

    def dot(self, a: str, b: str) -> str:
        return f"ct.matmul({a}, {b})"

    def scan(self, op: str, value: str) -> str:
        combine = {"max": ("ct.maximum", "float('-inf')"), "min": ("ct.minimum", "float('inf')")}.get(op)
        return f"ct.scan({value}, 0, {combine[0]}, {combine[1]})" if combine else super().scan(op, value)

    def atomic_add(self, base: str, at: Element, value: str, mask: str | None) -> str:
        if mask is not None:
            value = f"ct.where({mask}, {value}, 0)"
        return f"ct.atomic_add({base}, ({', '.join(at.indices)},), {value})"

    def launch(self, kernel: str, grid: list[str], arrays: list[str], symbols: list[str]) -> str:
        args = ", ".join(arrays + symbols)
        return f"ct.launch(cupy.cuda.get_current_stream(), ({', '.join(grid)},), {kernel}, ({args},))"
