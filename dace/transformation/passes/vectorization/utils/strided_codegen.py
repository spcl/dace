# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Render ``strided_load`` / ``strided_store`` intrinsic calls for CPU codegen.

Every compile-time-constant value is emitted as a template (non-type) argument
rather than a runtime function parameter: the lane count (``vector_width``) is
always constant, and the ``stride`` is a template argument too *when it is a
numeric constant*. A symbolic stride -- e.g. a multi-dim ``N`` that is an SDFG
symbol resolved only at runtime -- cannot be a template argument and is emitted
as the trailing runtime parameter (the header provides both overloads; see
``runtime/include/dace/cpu_vectorizable_math.h``).
"""
from typing import Optional, Union

import dace


def constant_stride(stride: Union[int, str, "dace.symbolic.SymbolicType"]) -> Optional[int]:
    """Return ``int(stride)`` when ``stride`` is a compile-time constant, else ``None``.

    A stride with free symbols (e.g. ``N`` or ``N + 1``) must stay a runtime
    argument; only a purely numeric stride can become a non-type template
    argument.
    """
    try:
        expr = dace.symbolic.pystr_to_symbolic(str(stride))
    except (TypeError, ValueError, AttributeError):
        return None
    if getattr(expr, "free_symbols", None):
        return None
    try:
        return int(expr)
    except (TypeError, ValueError):
        return None


def render_strided_call(fn: str,
                        dtype: str,
                        width: Union[int, str],
                        stride: Union[int, str, "dace.symbolic.SymbolicType"],
                        *,
                        masked: bool = False,
                        in_expr: str = "_in",
                        out_expr: str = "_out") -> str:
    """Build the C++ call string for a strided load/store intrinsic.

    :param fn: Intrinsic base name -- ``"strided_load"`` or ``"strided_store"``
        (``"_masked"`` is appended automatically when ``masked`` is set).
    :param dtype: C++ element type (e.g. ``"double"``).
    :param width: Lane count -- always a compile-time constant -> template arg.
    :param stride: Per-lane stride. Template arg when numeric, else a trailing
        runtime arg.
    :param masked: Emit the ``_masked`` variant with a trailing ``_mask`` runtime
        argument.
    :param in_expr: Source pointer expression (e.g. ``"_in"`` or ``"_in + 1"``).
    :param out_expr: Destination pointer expression.
    """
    name = f"{fn}_masked" if masked else fn
    width = int(width)
    mask_arg = ", _mask" if masked else ""
    cst = constant_stride(stride)
    if cst is not None:
        return f"{name}<{dtype}, {width}, {cst}>({in_expr}, {out_expr}{mask_arg});"
    return f"{name}<{dtype}, {width}>({in_expr}, {out_expr}, {stride}{mask_arg});"
