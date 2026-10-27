# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import SDFG, SDFGState, library, properties
from dace import memlet as mm
from dace.codegen.common import sym2cpp
from dace.frontend.common import op_repository as oprepo
from dace.libraries.blas import blas_helpers, gpu_dialect, vendor
from dace.libraries.blas.blas_helpers import operand
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandScalPure(ExpandTransformation):
    environments: list[type] = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = blas_helpers.connector_sdfg(node, parent_state, parent_sdfg)
        sdfg.add_state().add_mapped_tasklet(
            "scal",
            {"__i": f"0:{sdfg.arrays['_x'].shape[0]}"},
            {"__x": dace.Memlet("_x[__i]")},
            f"__res = ({node.a}) * __x",
            {"__res": dace.Memlet("_res[__i]")},
            external_edges=True,
        )
        return sdfg


def scal_call(node, state, sdfg, environment=None) -> SDFG:
    """``_res = _x``, then ``?scal`` scales ``_res`` in place, through CBLAS or the GPU library of ``environment``."""
    x, res = operand(node, state, sdfg, "_x"), operand(node, state, sdfg, "_res")
    dtype = x.desc.dtype
    if environment is not None:
        gpu = gpu_dialect.gpu_call(dtype)
        if gpu is None:
            return ExpandScalPure.expansion(node, state, sdfg)
        scalars, arg = gpu_dialect.scalar_pointers(gpu, dtype, a=node.a)
        call = f"{gpu.routine('scal')}({gpu_dialect.HANDLE}, {x.shape[0]}, {arg['a']}, {gpu.pointer('__res')}, {res.strides[0]})"
        code = gpu_dialect.call_code(gpu, environment, node, scalars, call)
        return blas_helpers.in_place_call(node, state, sdfg, "_x", "_res", code)
    prefix = blas_helpers.blas_prefix(dtype)
    if prefix is None:
        return ExpandScalPure.expansion(node, state, sdfg)
    scalars, arg = blas_helpers.scalar_arguments(dtype, a=node.a)
    call = f"cblas_{prefix.lower()}scal({sym2cpp(x.shape[0])}, {arg['a']}, __res, {sym2cpp(res.strides[0])});"
    return blas_helpers.in_place_call(node, state, sdfg, "_x", "_res", scalars + call)


@library.node
class Scal(nodes.LibraryNode):
    """BLAS ``?scal``: ``_res = a * _x`` for vectors."""

    implementations = {
        "pure": ExpandScalPure,
        **vendor.vendor_expansions("Scal", scal_call),
    }
    default_implementation = None

    a = properties.SymbolicProperty(allow_none=False, default=1, desc="The scaling factor.")

    def __init__(self, name, a=1, **kwargs):
        super().__init__(name, inputs=["_x"], outputs=["_res"], **kwargs)
        self.a = a

    def validate(self, sdfg, state):
        blas_helpers.require_one_dtype(self, state, sdfg, "_x", "_res")
        x, res = operand(self, state, sdfg, "_x").shape, operand(self, state, sdfg, "_res").shape
        if len(x) != 1 or x != res:
            raise ValueError(f"{self.label}: _x and _res must be vectors of one length")


@oprepo.replaces("dace.libraries.blas.scal")
@oprepo.replaces("dace.libraries.blas.Scal")
def scal_libnode(pv, sdfg: SDFG, state: SDFGState, a, x, result=None):
    result = result if result is not None else x
    node = Scal("scal", a=a)
    state.add_edge(state.add_read(x), None, node, "_x", mm.Memlet(x))
    state.add_edge(node, "_res", state.add_write(result), None, mm.Memlet(result))
    return []
