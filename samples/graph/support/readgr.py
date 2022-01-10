# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from ctypes import *
import os
import numpy as np

idx_t = c_uint32
real_t = c_float
size_t = c_uint64


class cnbr_t(Structure):
    _fields_ = [('pid', idx_t), ('ed', idx_t)]


class ckrinfo_t(Structure):
    _fields_ = [('id', idx_t), ('ed', idx_t), ('nnbrs', idx_t), ('inbr', idx_t)]


class vnbr_t(Structure):
    _fields_ = [('pid', idx_t), ('ned', idx_t), ('gv', idx_t)]


class vkrinfo_t(Structure):
    _fields_ = [('nid', idx_t), ('ned', idx_t), ('gv', idx_t), ('nnbrs', idx_t), ('inbr', idx_t)]


class nrinfo_t(Structure):
    _fields_ = [('edegrees', idx_t * 2)]


# yapf: disable
class graph_t (Structure):
    _fields_ = [
        ('nvtxs', idx_t),
        ('nedges', idx_t),
        ('ncon', idx_t),
        ('xadj', POINTER(idx_t)),
        ('vwgt', POINTER(idx_t)),
        ('vsize', POINTER(idx_t)),
        ('adjncy', POINTER(idx_t)),
        ('adjwgt', POINTER(idx_t)),
        ('tvwgt', POINTER(idx_t)),
        ('invtvwgt', POINTER(real_t)),
        ('readvw', c_bool),
        ('readew', c_bool),
        ('free_xadj', c_int),
        ('free_vwgt', c_int),
        ('free_vsize', c_int),
        ('free_adjncy', c_int),
        ('free_adjwgt', c_int),
        ('label', POINTER(idx_t)),
        ('cmap', POINTER(idx_t)),
        ('mincut', idx_t),
        ('minvol', idx_t),
        ('where', POINTER(idx_t)),
        ('pwgts', POINTER(idx_t)),
        ('nbnd', idx_t),
        ('bndptr', POINTER(idx_t)),
        ('bndind', POINTER(idx_t)),
        ('id', POINTER(idx_t)),
        ('ed', POINTER(idx_t)),
        ('ckrinfo', POINTER(ckrinfo_t)),
        ('vkrinfo', POINTER(vkrinfo_t)),
        ('nrinfo', POINTER(nrinfo_t)),
        ('coarser', c_void_p),
        ('finer', c_void_p)
    ]
# yapf: enable


def read_grfile(filename, with_weights=False):
    curpath = os.path.abspath(os.path.dirname(__file__))
    lib = CDLL(os.path.join(curpath, 'libreadgr.so'))
    lib.ReadGraphGR.restype = POINTER(graph_t)

    # Read graph
    graph = lib.ReadGraphGR(c_char_p(filename.encode('utf-8')))

    V = graph.contents.nvtxs
    E = graph.contents.nedges

    G_row = np.ctypeslib.as_array(graph.contents.xadj, shape=(V + 1, ))
    G_col = np.ctypeslib.as_array(graph.contents.adjncy, shape=(E, ))
    if with_weights:
        G_val = np.ctypeslib.as_array(graph.contents.adjwgt, shape=(E, ))

    # Do not free graph! numpy arrays are constructed from it
    #lib.FreeGraph(graph)

    if with_weights:
        return V, E, G_row, G_col, G_val
    else:
        return V, E, G_row, G_col
