# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Classes to handle Einstein-notation sums (einsum) as a library node. """
from functools import reduce
from itertools import chain
from string import ascii_letters
from typing import Dict, Optional

from dace import dtypes, symbolic
from dace.sdfg.nodes import AccessNode
from dace.sdfg import SDFG, SDFGState, InterstateEdge
from dace.memlet import Memlet
from dace.frontend.common import op_repository as oprepo


def _is_sequential(index_list):
    if not index_list:
        return True
    index_list = sorted(index_list)
    smallest_elem = index_list[0]
    return index_list == list(
        range(smallest_elem, smallest_elem + len(index_list)))


class EinsumParser(object):
    """ String parser for einsum. """
    def __init__(self, string):
        inout = string.split('->')
        if len(inout) == 1:
            inputs, output = string, ''
        else:
            inputs, output = inout

        for char in chain(inputs, output):
            if char not in ascii_letters + ',':
                raise ValueError(
                    'Invalid einsum string, subscript must contain'
                    ' letters, commas, and "->".')

        inputs = inputs.split(',')

        # No output given, assumed all "free" subscripts in inputs
        if len(inout) == 1:
            # Find intersection and union of all inputs for the non-outputs
            # and free inputs
            nonfree = set()
            free = set()
            for i, inp in enumerate(inputs):
                for var in set(inp):
                    if (all(var not in set(s) for s in inputs[i + 1:])
                            and var not in nonfree):
                        free.add(var)
                    else:
                        nonfree.add(var)
            output = ''.join(sorted(free))

        self.inputs = inputs
        self.output = output
        if len(inputs) != 2:
            return

        # Special case: contracting two tensors
        a, b = inputs
        c = output
        a_vars = set(a)
        b_vars = set(b)
        ab_vars = a_vars.union(b_vars)
        c_vars = set(c)
        if not ab_vars.issuperset(c_vars):
            raise ValueError('Einsum subscript string includes outputs that do'
                             ' not appear as an input')

        batch_vars = a_vars.intersection(b_vars).intersection(c_vars)
        sum_vars = a_vars.intersection(b_vars) - c_vars
        a_only_vars = a_vars - sum_vars - batch_vars
        b_only_vars = b_vars - sum_vars - batch_vars

        self.a_batch = [i for i, d in enumerate(a) if d in batch_vars]
        self.a_sum = [i for i, d in enumerate(a) if d in sum_vars]
        self.a_only = [i for i, d in enumerate(a) if d in a_only_vars]

        self.b_batch = [i for i, d in enumerate(b) if d in batch_vars]
        self.b_sum = [i for i, d in enumerate(b) if d in sum_vars]
        self.b_only = [i for i, d in enumerate(b) if d in b_only_vars]

        self.c_a_only = [i for i, d in enumerate(c) if d in a_only_vars]
        self.c_b_only = [i for i, d in enumerate(c) if d in b_only_vars]
        self.c_batch = [i for i, d in enumerate(c) if d in batch_vars]

    def is_bmm(self):
        if len(self.inputs) != 2:
            return False
        for key, val in self.fields().items():
            if not _is_sequential(val):
                return False
        return True

    def fields(self):
        return {
            fname: fval
            for fname, fval in self.__dict__.items()
            if fname not in ('inputs', 'output')
        }

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)


def create_batch_gemm_sdfg(dtype, strides):
    #########################
    sdfg = SDFG('einsum')
    state = sdfg.add_state()
    M, K, N = (symbolic.symbol(s) for s in ['M', 'K', 'N'])
    BATCH, sAM, sAK, sAB, sBK, sBN, sBB, sCM, sCN, sCB = (
        symbolic.symbol(s) if symbolic.issymbolic(strides[s]) else strides[s]
        for s in [
            'BATCH', 'sAM', 'sAK', 'sAB', 'sBK', 'sBN', 'sBB', 'sCM', 'sCN',
            'sCB'
        ])

    batched = strides['BATCH'] != 1

    _, xarr = sdfg.add_array(
        'X',
        dtype=dtype,
        shape=[BATCH, M, K] if batched else [M, K],
        strides=[sAB, sAM, sAK] if batched else [sAM, sAK])
    _, yarr = sdfg.add_array(
        'Y',
        dtype=dtype,
        shape=[BATCH, K, N] if batched else [K, N],
        strides=[sBB, sBK, sBN] if batched else [sBK, sBN])
    _, zarr = sdfg.add_array(
        'Z',
        dtype=dtype,
        shape=[BATCH, M, N] if batched else [M, N],
        strides=[sCB, sCM, sCN] if batched else [sCM, sCN])

    gX = state.add_read('X')
    gY = state.add_read('Y')
    gZ = state.add_write('Z')

    import dace.libraries.blas as blas  # Avoid import loop

    libnode = blas.MatMul('einsum_gemm', zarr.dtype)
    state.add_node(libnode)
    state.add_edge(gX, None, libnode, '_a', Memlet.from_array(gX.data, xarr))
    state.add_edge(gY, None, libnode, '_b', Memlet.from_array(gY.data, yarr))
    state.add_edge(libnode, '_c', gZ, None, Memlet.from_array(gZ.data, zarr))

    return sdfg


def prod(iterable):
    return reduce(lambda x, y: x * y, iterable, 1)


@oprepo.replaces('numpy.einsum')
def create_einsum_sdfg(sdfg: SDFG,
                       state: SDFGState,
                       einsum_string: str,
                       *arrays: str,
                       dtype: Optional[dtypes.typeclass] = None,
                       optimize: bool = False,
                       output: Optional[str] = None):
    return _create_einsum_internal(sdfg,
                                   state,
                                   einsum_string,
                                   *arrays,
                                   dtype=dtype,
                                   optimize=optimize,
                                   output=output)[0]


def _create_einsum_internal(sdfg: SDFG,
                            state: SDFGState,
                            einsum_string: str,
                            *arrays: str,
                            dtype: Optional[dtypes.typeclass] = None,
                            optimize: bool = False,
                            output: Optional[str] = None,
                            nodes: Optional[Dict[str, AccessNode]] = None):
    # Infer shapes and strides of input/output arrays
    einsum = EinsumParser(einsum_string)

    if len(einsum.inputs) != len(arrays):
        raise ValueError('Invalid number of arrays for einsum expression')

    # Get shapes from arrays and verify dimensionality
    chardict = {}
    for inp, inpname in zip(einsum.inputs, arrays):
        inparr = sdfg.arrays[inpname]
        if len(inp) != len(inparr.shape):
            raise ValueError('Dimensionality mismatch in input "%s"' % inpname)
        for char, shp in zip(inp, inparr.shape):
            if char in chardict and shp != chardict[char]:
                raise ValueError('Dimension mismatch in einsum expression')
            chardict[char] = shp

    if optimize:
        # Try to import opt_einsum
        try:
            import opt_einsum as oe
        except (ModuleNotFoundError, NameError, ImportError):
            raise ImportError('To optimize einsum expressions, please install '
                              'the "opt_einsum" package.')

        for char, shp in chardict.items():
            if symbolic.issymbolic(shp):
                raise ValueError('Einsum optimization cannot be performed '
                                 'on symbolically-sized array dimension "%s" '
                                 'for subscript character "%s"' % (shp, char))

        # Create optimal contraction path
        # noinspection PyTypeChecker
        _, path_info = oe.contract_path(
            einsum_string, *oe.helpers.build_views(einsum_string, chardict))

        input_nodes = nodes or {arr: state.add_read(arr) for arr in arrays}
        result_node = None

        # Follow path and create a chain of operation SDFG states
        for pair, nonfree, expr, after, blas in path_info.contraction_list:
            result, result_node = _create_einsum_internal(sdfg,
                                                          state,
                                                          expr,
                                                          arrays[pair[0]],
                                                          arrays[pair[1]],
                                                          dtype=dtype,
                                                          optimize=False,
                                                          output=None,
                                                          nodes=input_nodes)
            arrays = ([a for i, a in enumerate(arrays) if i not in pair] +
                      [result])
            input_nodes[result] = result_node

        return arrays[0], result_node
        # END of einsum optimization

    input_nodes = nodes or {arr: state.add_read(arr) for arr in arrays}

    # Get output shape from chardict, or [1] for a scalar output
    output_shape = list(map(lambda k: chardict[k], einsum.output)) or [1]
    output_index = ','.join(o for o in einsum.output) or '0'

    if output is None:
        dtype = dtype or sdfg.arrays[arrays[0]].dtype
        output, odesc = sdfg.add_temp_transient(output_shape, dtype)
        to_init = True
    else:
        odesc = sdfg.arrays[output]
        dtype = dtype or odesc.dtype
        to_init = False

    if not einsum.is_bmm():
        # Fall back to "pure" SDFG einsum with conflict resolution
        c = state.add_write(output)

        # Add state before this one to initialize the output value
        if to_init:
            init_state = sdfg.add_state_before(state)
            if len(einsum.output) > 0:
                init_state.add_mapped_tasklet(
                    'einsum_reset',
                    {k: '0:%s' % chardict[k]
                     for k in einsum.output}, {},
                    'out_%s = 0' % output,
                    {'out_%s' % output: Memlet.simple(output, output_index)},
                    external_edges=True)
            else:  # Scalar output
                t = init_state.add_tasklet('einsum_reset', set(),
                                           {'out_%s' % output},
                                           'out_%s = 0' % output)
                onode = init_state.add_write(output)
                init_state.add_edge(t, 'out_%s' % output, onode, None,
                                    Memlet.simple(output, '0'))

        # Pure einsum map
        state.add_mapped_tasklet(
            'einsum', {k: '0:%s' % v
                       for k, v in chardict.items()}, {
                           'inp_%s' % arr: Memlet.simple(arr, ','.join(inp))
                           for inp, arr in zip(einsum.inputs, arrays)
                       },
            'out_%s = %s' % (output, ' * '.join('inp_%s' % arr
                                                for arr in arrays)),
            {
                'out_%s' % output:
                Memlet.simple(output, output_index, wcr_str='lambda a,b: a+b')
            },
            input_nodes=input_nodes,
            output_nodes={output: c},
            external_edges=True)
    else:
        # Represent einsum as a GEMM or batched GEMM (using library nodes)
        a_shape = sdfg.arrays[arrays[0]].shape
        b_shape = sdfg.arrays[arrays[1]].shape
        c_shape = output_shape

        a = input_nodes[arrays[0]]
        b = input_nodes[arrays[1]]
        c = state.add_write(output)

        # Compute GEMM dimensions and strides
        strides = dict(
            BATCH=prod([c_shape[dim] for dim in einsum.c_batch]),
            M=prod([a_shape[dim] for dim in einsum.a_only]),
            K=prod([a_shape[dim] for dim in einsum.a_sum]),
            N=prod([b_shape[dim] for dim in einsum.b_only]),
            sAM=prod(a_shape[einsum.a_only[-1] + 1:]) if einsum.a_only else 1,
            sAK=prod(a_shape[einsum.a_sum[-1] + 1:]) if einsum.a_sum else 1,
            sAB=prod(a_shape[einsum.a_batch[-1] +
                             1:]) if einsum.a_batch else 1,
            sBK=prod(b_shape[einsum.b_sum[-1] + 1:]) if einsum.b_sum else 1,
            sBN=prod(b_shape[einsum.b_only[-1] + 1:]) if einsum.b_only else 1,
            sBB=prod(b_shape[einsum.b_batch[-1] +
                             1:]) if einsum.b_batch else 1,
            sCM=prod(c_shape[einsum.c_a_only[-1] +
                             1:]) if einsum.c_a_only else 1,
            sCN=prod(c_shape[einsum.c_b_only[-1] +
                             1:]) if einsum.c_b_only else 1,
            sCB=prod(c_shape[einsum.c_batch[-1] +
                             1:]) if einsum.c_batch else 1)

        # Complement strides to make matrices as necessary
        if len(a_shape) == 1 and len(einsum.a_sum) == 1:
            strides['sAK'] = 1
            strides['sAB'] = strides['sAM'] = strides['K']
        if len(b_shape) == 1 and len(einsum.b_sum) == 1:
            strides['sBN'] = 1
            strides['sBK'] = 1
            strides['sBB'] = strides['K']
        if len(c_shape) == 1 and len(einsum.a_sum) == len(einsum.b_sum):
            strides['sCN'] = 1
            strides['sCB'] = strides['sCM'] = strides['N']

        # Create nested SDFG for GEMM
        nsdfg = create_batch_gemm_sdfg(dtype, strides)

        nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'X', 'Y'}, {'Z'},
                                           strides)
        state.add_edge(a, None, nsdfg_node, 'X',
                       Memlet.from_array(a.data, a.desc(sdfg)))
        state.add_edge(b, None, nsdfg_node, 'Y',
                       Memlet.from_array(b.data, b.desc(sdfg)))
        state.add_edge(nsdfg_node, 'Z', c, None,
                       Memlet.from_array(c.data, c.desc(sdfg)))

    return output, c
