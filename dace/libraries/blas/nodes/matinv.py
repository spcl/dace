import copy
from typing import Any, Dict
from dace.subsets import Range
from dace.symbolic import symstr
from dace.properties import Property
import dace.library
import dace.sdfg.nodes
from dace.transformation.pattern_matching import ExpandTransformation
from dace.libraries.blas.blas_helpers import to_blastype, get_leading_dimension
from dace.libraries.blas.nodes import Getrf, Getri
from .. import environments
import numpy as np


def _make_sdfg(node, parent_state, parent_sdfg, implementation):

    inout_dict = node.validate(parent_sdfg, parent_state)

    sdfg = dace.SDFG(node.label + "_sdfg")
    state = sdfg.add_state(node.label + "_state")

    for k, v in inout_dict.items():
        sdfg.add_array(k, **v)

    node1 = Getrf('getrf_node', dtype=dace.float64)
    node1.implementation = implementation
    node2 = Getri('getri_node', dtype=dace.float64)
    node2.implementation = implementation
    a = state.add_read('_minv_a_out')
    b = state.add_access('_minv_a_out')
    c = state.add_write('_minv_a_out')
    ipiv = state.add_access('_minv_ipiv')
    info1 = state.add_write('_minv_info')
    info2 = state.add_write('_minv_info')

    a_size = inout_dict['_minv_a_in']['shape']
    a_subset = Range([(0, s-1, 1) for s in a_size])
    ipiv_size = inout_dict['_minv_ipiv']['shape']
    ipiv_subset = Range([0, s-1, 1] for s in ipiv_size)

    state.add_edge(a, None, node1, '_a_in',
                   dace.Memlet(data='_minv_a_out', subset=a_subset))
    state.add_edge(node1, '_a_out', b, None,
                   dace.Memlet(data='_minv_a_out', subset=a_subset))
    state.add_edge(node1, '_ipiv', ipiv, None,
                   dace.Memlet(data='_minv_ipiv', subset=ipiv_subset))
    state.add_edge(node1, '_info', info1, None,
                   dace.Memlet.simple('_minv_info', '0'))

    state.add_edge(b, None, node2, '_a_in',
                   dace.Memlet(data='_minv_a_out', subset=a_subset))
    state.add_edge(node2, '_a_out', c, None,
                   dace.Memlet(data='_minv_a_out', subset=a_subset))
    state.add_edge(ipiv, None, node2, '_ipiv',
                   dace.Memlet(data='_minv_ipiv', subset=ipiv_subset))
    state.add_edge(node2, '_info', info2, None,
                   dace.Memlet.simple('_minv_info', '0'))
    
    return sdfg


@dace.library.expansion
class ExpandMatInvPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        raise NotImplementedError("Missing pure implementation of GETRI.")

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandMatInvPure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandMatInvMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, state, sdfg):
        return _make_sdfg(node, state, sdfg, 'MKL')


@dace.library.node
class MatInv(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandMatInvPure,
        "MKL": ExpandMatInvMKL,
        # "cuBLAS": ExpandMatInvCuBLAS
    }
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)

    def __init__(self,
                 name,
                 dtype=None,
                 location=None):
        super().__init__(name,
                         location=location,
                         inputs={"_minv_a_in"},
                         outputs={"_minv_a_out", "_minv_ipiv", "_minv_info"})
        self.dtype = dtype

    def validate(self, sdfg, state):

        ret = dict()

        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Expected 1 input to MatInv")
        for edge in state.in_edges(self):
            _, _, _, dst_conn, memlet = edge
            if dst_conn == "_minv_a_in":
                srcname = memlet.data
                srcsubset = memlet.subset
                subset = copy.deepcopy(memlet.subset)
                squeezed = subset.squeeze()
                size_a_in = subset.size()
                array = sdfg.data(
                    dace.sdfg.find_input_arraynode(state, edge).data)
                strides = [
                    s for i, s in enumerate(array.strides) if i in squeezed
                ]
                ret['_minv_a_in'] = dict(shape=size_a_in, dtype=array.dtype,
                                         strides=strides, storage=array.storage)

        if len(size_a_in) != 2:
            raise ValueError("MatInv supported only on a matrix input A")

        if size_a_in[0] != size_a_in[1]:
            raise ValueError("MatInv supported only on square matrices")

        out_edges = state.out_edges(self)
        if len(out_edges) != 3:
            raise ValueError(
                "Expected 3 outputs from MatInv")
        for edge in state.out_edges(self):
            _, src_conn, _, _, memlet = edge
            if src_conn == "_minv_a_out":
                dstname = memlet.data
                dstsubset = memlet.subset
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_a_out = subset.size()
                array = sdfg.data(
                    dace.sdfg.find_output_arraynode(state, edge).data)
                strides = [
                    s for i, s in enumerate(array.strides) if i in squeezed
                ]
                ret['_minv_a_out'] = dict(shape=size_a_out, dtype=array.dtype,
                                         strides=strides, storage=array.storage)
            if src_conn == "_minv_ipiv":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_ipiv = subset.size()
                array = sdfg.data(
                    dace.sdfg.find_output_arraynode(state, edge).data)
                strides = [
                    s for i, s in enumerate(array.strides) if i in squeezed
                ]
                ret['_minv_ipiv'] = dict(shape=size_ipiv, dtype=array.dtype,
                                         strides=strides, storage=array.storage)
            if src_conn == "_minv_info":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_info = subset.size()
                array = sdfg.data(
                    dace.sdfg.find_output_arraynode(state, edge).data)
                strides = [
                    s for i, s in enumerate(array.strides) if i in squeezed
                ]
                ret['_minv_info'] = dict(shape=size_info, dtype=array.dtype,
                                         strides=strides, storage=array.storage)

        if srcname != dstname:
            raise ValueError("MatInv input and output must be the same matrix A")

        if len(size_a_out) != 2:
            raise ValueError("MatInv output must be a matrix")

        if srcsubset != dstsubset:
            raise ValueError("MatInv input and output subsets do not match")

        expected_size_ipiv = min(size_a_in)
        if size_ipiv[0] < expected_size_ipiv:
            raise ValueError("1D array for pivot indices must have length at N")

        return ret
