# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LAPACK ``?GETRI``: the inverse of a matrix from its ``getrf`` factors and pivots, in place (``_xin`` and
``_xout`` name one buffer).

Neither vendor GPU solver has a ``getri``; the device expansion solves ``A X = I`` with their ``getrs`` instead, so
it reads the factors column-major as their ``getrf`` leaves them. The thread-block expansion inverts in place.
"""

import copy

import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import Memlet, dtypes
from dace.libraries.blas import blas_helpers
from dace.libraries.blas import environments as blas_environments
from dace.libraries.lapack import block as lapack_block
from dace.libraries.lapack import environments
from dace.libraries.lapack.block import BLOCK_LAPACK, ExpandLapackBlock
from dace.libraries.lapack.nodes.getrs import Getrs
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES, call_scratch_lifetime, scratch_storage
from dace.libraries.standard.pure_components import operand_array
from dace.ordered import OrderedSet
from dace.transformation.transformation import ExpandTransformation


@dace.library.expansion
class ExpandGetriPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of LAPACK GETRI.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        raise (NotImplementedError)


@dace.library.expansion
class ExpandGetriOpenBLAS(ExpandTransformation):
    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x, rows_x, cols_x), desc_ipiv, desc_result = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        lapack_dtype = blas_helpers.to_blastype(dtype.type).lower()
        cast = ""
        if lapack_dtype == "c":
            cast = "(lapack_complex_float*)"
        elif lapack_dtype == "z":
            cast = "(lapack_complex_double*)"
        if desc_x.dtype.veclen > 1:
            raise (NotImplementedError)

        n = n or node.n
        code = f"_res = LAPACKE_{lapack_dtype}getri(LAPACK_ROW_MAJOR, {rows_x}, {cast}_xin, {stride_x}, _ipiv);"
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name, node.in_connectors, node.out_connectors, code, language=dace.dtypes.Language.CPP
        )
        return tasklet


@dace.library.expansion
class ExpandGetriMKL(ExpandTransformation):
    environments = [blas_environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x, rows_x, cols_x), desc_ipiv, desc_result = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        lapack_dtype = blas_helpers.to_blastype(dtype.type).lower()
        cast = ""
        if lapack_dtype == "c":
            cast = "(MKL_Complex8*)"
        elif lapack_dtype == "z":
            cast = "(MKL_Complex16*)"
        if desc_x.dtype.veclen > 1:
            raise (NotImplementedError)

        n = n or node.n
        code = f"_res = LAPACKE_{lapack_dtype}getri(LAPACK_ROW_MAJOR, {rows_x}, {cast}_xin, {stride_x}, _ipiv);"
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name, node.in_connectors, node.out_connectors, code, language=dace.dtypes.Language.CPP
        )
        return tasklet


class ExpandGetriGPUSolver(ExpandTransformation):
    """``A^-1`` on a vendor GPU solver: ``getrs`` of the factors against an identity scratch, copied back."""

    environments = []
    solver = ""

    @classmethod
    def expansion(cls, node, parent_state, parent_sdfg, **kwargs):
        desc_x, n = node.validate(parent_sdfg, parent_state)[0][::2]
        sdfg = dace.SDFG(f"{node.label}_sdfg")
        edges = {e.dst_conn: e for e in parent_state.in_edges(node)}
        edges.update({e.src_conn: e for e in parent_state.out_edges(node)})
        arrays = {
            conn: operand_array(sdfg, conn, edges[conn], parent_sdfg) for conn in ("_xin", "_ipiv", "_xout", "_res")
        }
        lifetime = call_scratch_lifetime(node, parent_state, parent_sdfg, n)
        eye_desc = sdfg.add_array(
            "_eye",
            [n, n],
            desc_x.dtype,
            transient=True,
            storage=scratch_storage(desc_x.storage, lifetime),
            lifetime=lifetime,
        )[1]
        state = sdfg.add_state(f"{node.label}_state")
        on_device = desc_x.storage in GPU_RESIDENT_STORAGES
        eye_exit = state.add_mapped_tasklet(
            "_eye_",
            dict(__i0=f"0:{n}", __i1=f"0:{n}"),
            {},
            "_out = (__i0 == __i1) ? 1 : 0;",
            dict(_out=Memlet("_eye[__i0, __i1]")),
            language=dtypes.Language.CPP,
            schedule=dtypes.ScheduleType.GPU_Device if on_device else dtypes.ScheduleType.Default,
            external_edges=True,
        )[2]
        solve = Getrs("getrs")
        solve.implementation = cls.solver
        solved = state.add_access("_eye")
        state.add_edge(state.add_read("_xin"), None, solve, "_a", Memlet.from_array("_xin", arrays["_xin"]))
        state.add_edge(state.add_read("_ipiv"), None, solve, "_ipiv", Memlet.from_array("_ipiv", arrays["_ipiv"]))
        state.add_edge(state.out_edges(eye_exit)[0].dst, None, solve, "_rhs_in", Memlet.from_array("_eye", eye_desc))
        state.add_edge(solve, "_rhs_out", solved, None, Memlet.from_array("_eye", eye_desc))
        state.add_edge(solve, "_res", state.add_write("_res"), None, Memlet.from_array("_res", arrays["_res"]))
        state.add_nedge(solved, state.add_write("_xout"), Memlet.from_array("_eye", eye_desc))
        return sdfg


@dace.library.expansion
class ExpandGetriCuSolverDn(ExpandGetriGPUSolver):
    environments = [environments.cusolverdn.cuSolverDn]
    solver = "cuSolverDn"


@dace.library.expansion
class ExpandGetriHipSolverDn(ExpandGetriGPUSolver):
    environments = [environments.hipsolverdn.hipSolverDn]
    solver = "hipSolverDn"


@dace.library.expansion
class ExpandGetriBlock(ExpandLapackBlock):
    """In-kernel inverse by one thread block (``block_lapack::getri``): invert U, solve ``X L = U^-1``, swap."""

    inverts = True

    @classmethod
    def body(cls, node, ops, uid):
        return (
            lapack_block.copy_code(ops, "_xin", "_xout")
            + f"const int __info = dace::block_lapack::getri({lapack_block.mat(ops, '_xout')}, {lapack_block.order(ops, '_xin')}, "
            + f"{lapack_block.pointer('_ipiv', True)}, {lapack_block.cpp(ops['_ipiv'].strides[0])});\n"
            + lapack_block.info_code(ops, "__info")
        )


@dace.library.node
class Getri(dace.sdfg.nodes.LibraryNode):
    # Global properties
    implementations = {
        "OpenBLAS": ExpandGetriOpenBLAS,
        "MKL": ExpandGetriMKL,
        "cuSolverDn": ExpandGetriCuSolverDn,
        "hipSolverDn": ExpandGetriHipSolverDn,
        BLOCK_LAPACK: ExpandGetriBlock,
    }
    default_implementation = None

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None, category="Semantics")

    def __init__(self, name, n=None, *args, **kwargs):
        super().__init__(
            name, *args, inputs=OrderedSet(("_xin", "_ipiv")), outputs=OrderedSet(("_xout", "_res")), **kwargs
        )

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (x, ipiv, res) of the three data descriptors in the
                 parent SDFG.
        """
        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to getri")
        out_edges = state.out_edges(self)
        if len(out_edges) != 2:
            raise ValueError("Expected exactly two outputs from getri")

        desc_xin, desc_xout, desc_ipiv, desc_res = None, None, None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_xin":
                desc_xin = sdfg.arrays[e.data.data]
            if e.dst_conn == "_ipiv":
                desc_ipiv = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_xout":
                desc_xout = sdfg.arrays[e.data.data]
            if e.src_conn == "_result":
                desc_res = sdfg.arrays[e.data.data]

        if desc_xin.dtype.base_type != desc_xout.dtype.base_type:
            raise ValueError("Basetype of input and output must be equal!")
        if desc_ipiv.dtype.base_type != dace.dtypes.int32:
            raise ValueError("Pivot input must be an integer array!")

        in_memlets = [None] * 2
        for _, _, _, conn, data in in_edges:
            if conn == "_xin":
                in_memlets[0] = data
            elif conn == "_ipiv":
                in_memlets[1] = data

        # Squeeze input memlets
        squeezed_xin = copy.deepcopy(in_memlets[0].subset)
        dims_xin = squeezed_xin.squeeze()
        squeezed_ipiv = copy.deepcopy(in_memlets[1].subset)
        dims_ipiv = squeezed_ipiv.squeeze()

        stride_x = desc_xin.strides[dims_xin[0]]
        shape_xin = squeezed_xin.size()
        rows_x = shape_xin[0]
        cols_x = shape_xin[1]

        return (desc_xin, stride_x, rows_x, cols_x), desc_ipiv, desc_res
