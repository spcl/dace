from dace.sdfg.propagation import propagate_memlets_sdfg, propagate_memlets_scope
from dace.sdfg.nodes import EntryNode, ExitNode
from dace.symbolic import pystr_to_symbolic
from dace.sdfg import SDFG
import numpy as np
import os, sys

from .util import expand_maps, expand_reduce, fusion, stencil_tiling

DATATYPE = np.float32
PATH = os.path.expanduser('~/dace/dace/transformation/estimator/programs')


def get_program(program_name):
    '''
    returns a post-processed SDFG of the given program
    as well as the graph of interest for analysis
    '''
    data_suffix = None
    if DATATYPE == np.float32:
        data_suffix = str(32)
    elif DATATYPE == np.float64:
        data_suffix = str(64)
    else:
        raise NotImplementedError("Wrong Datatype")

    if program_name == 'synthetic':
        sdfg = SDFG.from_file(
            os.path.join(PATH, 'synthetic' + data_suffix + '.sdfg'))
    elif program_name == 'vadv':
        sdfg = SDFG.from_file(os.path.join(PATH,
                                           'vadv' + data_suffix + '.sdfg'))
        # only propagate memlets at outermost maps, else propagation fails

        graph = sdfg.nodes()[0]
        # first full pass
        propagate_memlets_sdfg(sdfg)
        for node in graph.nodes():
            if isinstance(node, EntryNode):
                for e in graph.out_edges(node):
                    if e.data.dynamic:
                        e.data.dynamic = False
                        e.data.volume = pystr_to_symbolic('K-2')
            if isinstance(node, ExitNode):
                for e in graph.in_edges(node):
                    if e.data.dynamic:
                        e.data.dynamic = False
                        e.data.volume = pystr_to_symbolic('K-2')

        propagate_memlets_scope(sdfg, graph, graph.scope_leaves())
        sdfg.save('vadv_improved.sdfg')

    elif program_name == 'hdiff':
        sdfg = SDFG.from_file(
            os.path.join(PATH, 'hdiff' + data_suffix + '.sdfg'))

        propagate_memlets_sdfg(sdfg)
    elif program_name == 'hdiff_mini':
        sdfg = SDFG.from_file(
            os.path.join(PATH, 'hdiff_mini' + data_suffix + '.sdfg'))
        # fix the memlet mess
        propagate_memlets_sdfg(sdfg)
        print("*****", sdfg.constants)
    elif program_name == 'softmax':
        sdfg = SDFG.from_file(
            os.path.join(PATH, 'softmax' + data_suffix + '.sdfg'))
    elif program_name == 'gemver':
        sdfg = SDFG.from_file(
            os.path.join(PATH, 'gemver' + data_suffix + '.sdfg'))
    elif program_name == 'transformer':
        sdfg = SDFG.from_file(
            os.path.join(PATH, 'transformer' + data_suffix + '.sdfg'))
    else:
        raise NotImplementedError("Program not found")

    return sdfg


def get_args(program_name):
    '''
    returns a triple of dicts (inputs, ouputs, symbols)
    that serve as args for a given program
    '''
    if program_name == 'synthetic':
        N, M, O = 50, 60, 70
        return ({
            'A': np.random.rand(N).astype(DATATYPE),
            'B': np.random.rand(M).astype(DATATYPE),
            'C': np.random.rand(O).astype(DATATYPE)
        }, {
            'out1': np.zeros((N, M), DATATYPE),
            'out2': np.zeros((1), DATATYPE),
            'out3': np.zeros((N, M, O))
        }, {
            'N': N,
            'M': M,
            'O': O
        })
    elif program_name == 'softmax':
        H, B, SN, SM = 16, 16, 256, 256
        return ({
            'X_in': np.random.rand(H, B, SN, SM).astype(DATATYPE)
        }, {}, {
            'H': H,
            'B': B,
            'SN': SN,
            'SM': SM
        })

    elif program_name == 'vadv':
        I, J, K = 128, 128, 80
        wcon = np.random.rand(I + 1, J, K).astype(DATATYPE)
        u_stage = np.random.rand(I, J, K).astype(DATATYPE)
        utens_stage = np.random.rand(I, J, K).astype(DATATYPE)
        u_pos = np.random.rand(I, J, K).astype(DATATYPE)
        utens = np.random.rand(I, J, K).astype(DATATYPE)

        # define strides
        strides = {}
        arrays = ['utens_stage', 'u_stage', 'wcon', 'u_pos', 'utens']
        dimtuple = (0, 2, 1)
        for array in arrays:
            istride = f"_{array}_I_stride"
            jstride = f"_{array}_J_stride"
            kstride = f"_{array}_K_stride"
            stride_names = [istride, jstride, kstride]

            s = 1
            for i in reversed(dimtuple):
                strides[stride_names[i]] = s
                s *= (I, J, K)[i]

        return ({
            'wcon': wcon,
            'u_stage': u_stage,
            'utens_stage': utens_stage,
            'u_pos': u_pos,
            'utens': utens
        }, {
            'utens_stage': utens_stage
        }, {
            '_gt_loc__dtr_stage': np.float32(1.4242424),
            'I': I,
            'J': J,
            'K': K,
            **strides
        })

    elif program_name == 'hdiff_mini':
        I, J, K = 128, 128, 80
        halo = 4
        return (
            {
                'pp_in': np.random.rand(J, K + 1, I).astype(DATATYPE),
                'u_in': np.random.rand(J, K + 1, I).astype(DATATYPE),
                'crlato': np.random.rand(J).astype(DATATYPE),
                'crlatu': np.random.rand(J).astype(DATATYPE),
                'hdmask': np.random.rand(J, K + 1, I).astype(DATATYPE),
                'w_in': np.random.rand(J, K + 1, I).astype(DATATYPE),
                'v_in': np.random.rand(J, K + 1, I).astype(DATATYPE),
                # needs symbols as well due to some glitch in the sdfg lol
                'I': I,
                'J': J,
                'K': K,
                'halo': halo
            },
            {
                'pp': np.zeros([J, K + 1, I], dtype=DATATYPE),
                'w': np.zeros([J, K + 1, I], dtype=DATATYPE),
                'v': np.zeros([J, K + 1, I], dtype=DATATYPE),
                'u': np.zeros([J, K + 1, I], dtype=DATATYPE)
            },
            {
                'I': I,
                'J': J,
                'K': K,
                'halo': halo
            })

    elif program_name == 'hdiff':
        I, J, K = 128, 128, 80
        halo = 4

        return (
            {
                'pp_in': np.random.rand(J, K, I).astype(DATATYPE),
                'u_in': np.random.rand(J, K, I).astype(DATATYPE),
                'crlato': np.random.rand(J).astype(DATATYPE),
                'crlatu': np.random.rand(J).astype(DATATYPE),
                'acrlat0': np.random.rand(J).astype(DATATYPE),
                'crlavo': np.random.rand(J).astype(DATATYPE),
                'crlavu': np.random.rand(J).astype(DATATYPE),
                'hdmask': np.random.rand(J, K, I).astype(DATATYPE),
                'w_in': np.random.rand(J, K, I).astype(DATATYPE),
                'v_in': np.random.rand(J, K, I).astype(DATATYPE),
                # re-add symbols
                'I': I,
                'J': J,
                'K': K,
                'halo': halo
            },
            {
                'pp_out': np.zeros([J, K, I], dtype=DATATYPE),
                'w_out': np.zeros([J, K, I], dtype=DATATYPE),
                'v_out': np.zeros([J, K, I], dtype=DATATYPE),
                'u_out': np.zeros([J, K, I], dtype=DATATYPE)
            },
            {
                'I': I,
                'J': J,
                'K': K,
                'halo': halo
            })

    elif program_name == 'gemver':
        N = 100
        A = np.random.rand(N, N).astype(DATATYPE)
        u1 = np.random.rand(N).astype(DATATYPE)
        u2 = np.random.rand(N).astype(DATATYPE)
        v1 = np.random.rand(N).astype(DATATYPE)
        v2 = np.random.rand(N).astype(DATATYPE)
        y = np.random.rand(N).astype(DATATYPE)
        w = np.zeros([N], dtype=DATATYPE)
        alpha = np.random.rand(1).astype(DATATYPE)
        beta = np.random.rand(1).astype(DATATYPE)

        return ({
            'A': A,
            'u1': u1,
            'u2': u2,
            'v1': v1,
            'v2': v2,
            'y': y,
            'alpha': alpha,
            'beta': beta
        }, {
            'A': A
        }, {
            'N': N
        })

    elif program_name == 'transformer':
        raise NotImplementedError("TODO")
    else:
        raise NotImplementedError("Program not found")
