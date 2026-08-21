# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import warnings

import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas import blas_helpers
from .. import environments
from dace import memlet as mm, SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo


@dace.library.expansion
class ExpandScalPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of SCAL.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), (desc_res, stride_res), sz = node.validate(parent_sdfg, parent_state)
        n = n or node.n or sz
        a = node.a

        sdfg = dace.SDFG(node.label + "_sdfg")
        sdfg.add_array("_x", [n], desc_x.dtype, strides=[stride_x], storage=desc_x.storage)
        sdfg.add_array("_res", [n], desc_res.dtype, strides=[stride_res], storage=desc_res.storage)

        state = sdfg.add_state(node.label + "_state")
        state.add_mapped_tasklet(
            "scal",
            {"__i": f"0:{n}"},
            {"__xin": dace.Memlet("_x[__i]")},
            f"__xout = ({a}) * __xin",
            {"__xout": dace.Memlet("_res[__i]")},
            external_edges=True,
        )
        return sdfg


@dace.library.expansion
class ExpandScalOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), (desc_res, stride_res), sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandScalPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        prefix = func.lower()
        n = n or node.n or sz
        a = node.a

        # cBLAS scal updates in place; copy _x into _res first, then call.
        if dtype in (dace.complex64, dace.complex128):
            code = f"""
            {dtype.ctype} __alpha = {dtype.ctype}({a});
            cblas_{prefix}copy({n}, _x, {stride_x}, _res, {stride_res});
            cblas_{prefix}scal({n}, &__alpha, _res, {stride_res});
            """
        else:
            code = f"""
            cblas_{prefix}copy({n}, _x, {stride_x}, _res, {stride_res});
            cblas_{prefix}scal({n}, ({dtype.ctype})({a}), _res, {stride_res});
            """

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandScalMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandScalOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandScalCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), (desc_res, stride_res), sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandScalPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        n = n or node.n or sz
        a = node.a

        code = environments.cublas.cuBLAS.handle_setup_code(node)
        code += f"""
        {dtype.ctype} __alpha = {dtype.ctype}({a});
        cublas{func}copy(__dace_cublas_handle, {n}, _x, {stride_x}, _res, {stride_res});
        cublas{func}scal(__dace_cublas_handle, {n}, &__alpha, _res, {stride_res});
        """

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Scal(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandScalPure,
        "OpenBLAS": ExpandScalOpenBLAS,
        "MKL": ExpandScalMKL,
        "cuBLAS": ExpandScalCuBLAS,
    }
    default_implementation = None

    # Object fields
    a = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("a"))
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, a=None, n=None, **kwargs):
        super().__init__(name, inputs={"_x"}, outputs={"_res"}, **kwargs)
        self.a = a if a is not None else dace.symbolic.symbol("a")
        self.n = n

    def validate(self, sdfg, state):
        """
        :return: A three-tuple ((x, stride_x), (res, stride_res), n).
        """
        return blas_helpers.validate_level1_vector_to_vector(self, sdfg, state, "SCAL")


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.scal')
@oprepo.replaces('dace.libraries.blas.Scal')
def scal_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, a, x, result=None):
    result = result if result is not None else x
    x_in = state.add_read(x)
    res_out = state.add_write(result)

    libnode = Scal('scal', a=a, n=sdfg.arrays[x].shape[0])
    state.add_node(libnode)

    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(libnode, '_res', res_out, None, mm.Memlet(result))

    return []
