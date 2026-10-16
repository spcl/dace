# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Triton: a kernel takes pointers, a tile is a ``tl.tensor`` over ``tl.arange`` lanes."""

from dace import dtypes
from dace.codegen.python.kernel import Dialect, Element

#: Imported by the generated module: Triton, its math library, the kernel spellings of the SymPy helpers and the
#: pointer a CuPy array passes as (Triton takes any object with ``data_ptr()`` and a ``dtype``).
IMPORTS = """\
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def tile_int_floor(a, b):
    return (a - ((a % b) + b) % b) // b


@triton.jit
def tile_int_ceil(a, b):
    return -tile_int_floor(-a, b)


@triton.jit
def tile_multiply(a, b):
    return a * b


@triton.jit
def tile_maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def tile_minimum(a, b):
    return tl.minimum(a, b)


class TritonPointer:
    __slots__ = ("pointer", "dtype")

    def __init__(self, array):
        self.pointer, self.dtype = array.data.ptr, array.dtype

    def data_ptr(self):
        return self.pointer
"""


class Triton(Dialect):
    imports = IMPORTS
    decorator = "@triton.jit"
    module = "tl"
    functions = {
        "int_floor": "tile_int_floor",
        "int_ceil": "tile_int_ceil",
        "Min": "tl.minimum",
        "Max": "tl.maximum",
        "Abs": "tl.abs",
        "ipow": "libdevice.pow",
        "pow": "libdevice.pow",
        "fmod": "libdevice.fmod",
        "atan2": "libdevice.atan2",
        **{name: f"libdevice.{name}" for name in ("tan", "asin", "acos", "atan", "sinh", "cosh", "tanh")},
    }

    def program_id(self, axis: int) -> str:
        return f"tl.program_id({axis})"

    def lane(self, widths: list[int], dim: int) -> str:
        lanes = f"tl.arange(0, {widths[dim]})"
        if len(widths) == 1:
            return lanes
        return f"{lanes}[{', '.join(':' if k == dim else 'None' for k in range(len(widths)))}]"

    def load(self, base: str, at: Element, mask: str | None, dtype: dtypes.typeclass) -> str:
        pointers = f"{base} + {at.flat()}"
        return f"tl.load({pointers})" if mask is None else f"tl.load({pointers}, mask={mask}, other=0)"

    def store(self, base: str, at: Element, value: str, mask: str | None) -> str:
        masked = "" if mask is None else f", mask={mask}"
        return f"tl.store({base} + {at.flat()}, {value}{masked})"

    def cast_type(self, dtype: dtypes.typeclass) -> str:
        return f"tl.{ {dtypes.bool_: 'int1'}.get(dtype, dtype.to_string()) }"

    def cast(self, value: str, dtype: dtypes.typeclass) -> str:
        return f"tl.cast({value}, {self.cast_type(dtype)})"

    def full(self, widths: list[int], value: str, dtype: dtypes.typeclass) -> str:
        return f"tl.full({list(widths)}, {value}, {self.cast_type(dtype)})"

    def dot(self, a: str, b: str) -> str:
        return f"tl.dot({a}, {b})"

    def reduce(self, op: str, value: str, axis: int | None, rank: int) -> str:
        if op == "*":
            return f"tl.reduce({value}, {axis}, tile_multiply)"
        return super().reduce(op, value, axis, rank)

    def scan(self, op: str, value: str) -> str:
        combine = {"*": "tile_multiply", "max": "tile_maximum", "min": "tile_minimum"}.get(op)
        return f"tl.associative_scan({value}, 0, {combine})" if combine else super().scan(op, value)

    def atomic_add(self, base: str, at: Element, value: str, mask: str | None) -> str:
        masked = "" if mask is None else f", mask={mask}"
        return f"tl.atomic_add({base} + {at.flat()}, {value}{masked})"

    def launch(self, kernel: str, grid: list[str], arrays: list[str], symbols: list[str]) -> str:
        args = [f"TritonPointer({name})" for name in arrays] + symbols
        return f"{kernel}[({', '.join(grid)},)]({', '.join(args)})"
