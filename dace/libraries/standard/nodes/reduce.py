# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" File defining the reduction library node. """

import ast
from copy import deepcopy as dcpy
import dace
import itertools
import functools
import dace.serialize
import dace.library
from typing import Any, Dict, Set
from dace.config import Config
from dace.sdfg import SDFG, SDFGState, devicelevel_block_size, propagation
from dace.sdfg import graph
from dace.frontend.python.astutils import unparse
from dace.properties import (Property, CodeProperty, LambdaProperty, RangeProperty, DebugInfoProperty, SetProperty,
                             make_properties, indirect_properties, DataProperty, SymbolicProperty, ListProperty,
                             SDFGReferenceProperty, DictProperty, LibraryImplementationProperty)
from dace.frontend.operations import detect_reduction_type
from dace import data, subsets as sbs, dtypes
from dace import registry, subsets
import pydoc
import warnings
from dace.sdfg import nodes
from dace.transformation import transformation as pm
from dace.symbolic import symstr, issymbolic
from dace.libraries.standard.environments.cuda import CUDA


@dace.library.expansion
class ExpandReducePure(pm.ExpandTransformation):
    """
        Pure SDFG Reduce expansion replaces a reduce node with nested maps and
        edges with WCR.
    """
    environments = []

    @staticmethod
    def expansion(node: 'Reduce', state: SDFGState, sdfg: SDFG):
        node.validate(sdfg, state)
        inedge: graph.MultiConnectorEdge = state.in_edges(node)[0]
        outedge: graph.MultiConnectorEdge = state.out_edges(node)[0]
        insubset = dcpy(inedge.data.subset)
        isqdim = insubset.squeeze()
        outsubset = dcpy(outedge.data.subset)
        osqdim = outsubset.squeeze()
        input_dims = len(insubset)
        output_dims = len(outsubset)
        input_data = sdfg.arrays[inedge.data.data]
        output_data = sdfg.arrays[outedge.data.data]

        if len(osqdim) == 0:  # Fix for scalars
            osqdim = [0]

        # Standardize and squeeze axes
        axes = node.axes if node.axes is not None else [i for i in range(len(inedge.data.subset))]
        axes = [axis for axis in axes if axis in isqdim]

        # Create nested SDFG
        nsdfg = SDFG('reduce')

        nsdfg.add_array('_in',
                        insubset.size(),
                        input_data.dtype,
                        strides=[s for i, s in enumerate(input_data.strides) if i in isqdim],
                        storage=input_data.storage)

        nsdfg.add_array('_out',
                        outsubset.size(),
                        output_data.dtype,
                        strides=[s for i, s in enumerate(output_data.strides) if i in osqdim],
                        storage=output_data.storage)

        # Rename outer connectors and add to node
        inedge._dst_conn = '_in'
        outedge._src_conn = '_out'
        node.add_in_connector('_in')
        node.add_out_connector('_out')

        if len(axes) == 0:
            # Degenerate reduction, do nothing
            nstate = nsdfg.add_state()
            r = nstate.add_read('_in')
            w = nstate.add_write('_out')
            nstate.add_edge(
                r, None, w, None,
                dace.Memlet(data='_in',
                            subset=dace.subsets.Range.from_array(nsdfg.arrays['_in']),
                            other_subset=dace.subsets.Range.from_array(nsdfg.arrays['_out'])))
            return nsdfg

        # If identity is defined, add an initialization state
        if node.identity is not None:
            init_state = nsdfg.add_state()
            nstate = nsdfg.add_state()
            nsdfg.add_edge(init_state, nstate, dace.InterstateEdge())

            # Add initialization as a map
            init_state.add_mapped_tasklet(
                'reduce_init', {'_o%d' % i: '0:%s' % symstr(d)
                                for i, d in enumerate(outedge.data.subset.size())}, {},
                '__out = %s' % node.identity,
                {'__out': dace.Memlet.simple('_out', ','.join(['_o%d' % i for i in range(output_dims)]))},
                external_edges=True)
        else:
            nstate = nsdfg.add_state()
        # END OF INIT

        # (If axes != all) Add outer map, which corresponds to the output range
        if len(axes) != input_dims:
            # Interleave input and output axes to match input memlet
            ictr, octr = 0, 0
            input_subset = []
            for i in isqdim:
                if i in axes:
                    input_subset.append('_i%d' % ictr)
                    ictr += 1
                else:
                    input_subset.append('_o%d' % octr)
                    octr += 1

            ome, omx = nstate.add_map('reduce_output',
                                      {'_o%d' % i: '0:%s' % symstr(sz)
                                       for i, sz in enumerate(outsubset.size())})
            outm = dace.Memlet.simple('_out', ','.join(['_o%d' % i for i in range(output_dims)]), wcr_str=node.wcr)
            inmm = dace.Memlet.simple('_in', ','.join(input_subset))
        else:
            ome, omx = None, None
            outm = dace.Memlet.simple('_out', '0', wcr_str=node.wcr)
            inmm = dace.Memlet.simple('_in', ','.join(['_i%d' % i for i in range(len(axes))]))

        # Add inner map, which corresponds to the range to reduce, containing
        # an identity tasklet
        ime, imx = nstate.add_map(
            'reduce_values',
            {'_i%d' % i: '0:%s' % symstr(insubset.size()[isqdim.index(axis)])
             for i, axis in enumerate(sorted(axes))})

        # Add identity tasklet for reduction
        t = nstate.add_tasklet('identity', {'__inp'}, {'__out'}, '__out = __inp')

        # Connect everything
        r = nstate.add_read('_in')
        w = nstate.add_read('_out')
        if ome:
            nstate.add_memlet_path(r, ome, ime, t, dst_conn='__inp', memlet=inmm)
            nstate.add_memlet_path(t, imx, omx, w, src_conn='__out', memlet=outm)
        else:
            nstate.add_memlet_path(r, ime, t, dst_conn='__inp', memlet=inmm)
            nstate.add_memlet_path(t, imx, w, src_conn='__out', memlet=outm)

        from dace.transformation import dataflow
        nsdfg.apply_transformations_repeated(dataflow.MapCollapse)

        return nsdfg


@dace.library.expansion
class ExpandReducePureSequentialDim(pm.ExpandTransformation):
    """
        Pure SDFG Reduce expansion replaces a reduce node with nested maps and
        edges with WCR.
    """
    environments = []

    @staticmethod
    def expansion(node: 'Reduce', state: SDFGState, sdfg: SDFG):
        node.validate(sdfg, state)
        inedge: graph.MultiConnectorEdge = state.in_edges(node)[0]
        outedge: graph.MultiConnectorEdge = state.out_edges(node)[0]
        insubset = dcpy(inedge.data.subset)
        isqdim = insubset.squeeze()
        outsubset = dcpy(outedge.data.subset)
        osqdim = outsubset.squeeze()
        input_dims = len(insubset)
        output_dims = len(outsubset)
        input_data = sdfg.arrays[inedge.data.data]
        output_data = sdfg.arrays[outedge.data.data]

        if len(osqdim) == 0:  # Fix for scalars
            osqdim = [0]

        # Standardize and squeeze axes
        axes = node.axes if node.axes is not None else [i for i in range(len(inedge.data.subset))]
        axes = [axis for axis in axes if axis in isqdim]

        if not axes:  # Degenerate reduction
            return ExpandReducePure.expansion(node, state, sdfg)

        assert node.identity is not None

        # Create nested SDFG
        nsdfg = SDFG('reduce')

        nsdfg.add_array('_in',
                        insubset.size(),
                        input_data.dtype,
                        strides=[s for i, s in enumerate(input_data.strides) if i in isqdim],
                        storage=input_data.storage)

        nsdfg.add_array('_out',
                        outsubset.size(),
                        output_data.dtype,
                        strides=[s for i, s in enumerate(output_data.strides) if i in osqdim],
                        storage=output_data.storage)

        nsdfg.add_transient('acc', [1], nsdfg.arrays['_in'].dtype, dtypes.StorageType.Register)

        nstate = nsdfg.add_state()

        # Interleave input and output axes to match input memlet
        ictr, octr = 0, 0
        input_subset = []
        for i in isqdim:
            if i in axes:
                input_subset.append('_i%d' % ictr)
                ictr += 1
            else:
                input_subset.append('_o%d' % octr)
                octr += 1

        ome, omx = nstate.add_map('reduce_output',
                                  {'_o%d' % i: '0:%s' % symstr(sz)
                                   for i, sz in enumerate(outsubset.size())})
        outm = dace.Memlet.simple('_out', ','.join(['_o%d' % i for i in range(output_dims)]))
        #wcr_str=node.wcr)
        inmm = dace.Memlet.simple('_in', ','.join(input_subset))

        idt = nstate.add_tasklet('reset', {}, {'o'}, f'o = {node.identity}')
        nstate.add_edge(ome, None, idt, None, dace.Memlet())

        accread = nstate.add_access('acc')
        accwrite = nstate.add_access('acc')
        nstate.add_edge(idt, 'o', accread, None, dace.Memlet('acc'))

        # Add inner map, which corresponds to the range to reduce, containing
        # an identity tasklet
        ime, imx = nstate.add_map(
            'reduce_values',
            {'_i%d' % i: '0:%s' % symstr(insubset.size()[isqdim.index(axis)])
             for i, axis in enumerate(sorted(axes))},
            schedule=dtypes.ScheduleType.Sequential)

        # Add identity tasklet for reduction
        t = nstate.add_tasklet('identity', {'a', 'b'}, {'o'}, 'o = b')

        # Connect everything
        r = nstate.add_read('_in')
        w = nstate.add_write('_out')
        nstate.add_memlet_path(r, ome, ime, t, dst_conn='b', memlet=inmm)
        nstate.add_memlet_path(accread, ime, t, dst_conn='a', memlet=dace.Memlet('acc[0]'))
        nstate.add_memlet_path(t, imx, accwrite, src_conn='o', memlet=dace.Memlet('acc[0]', wcr=node.wcr))
        nstate.add_memlet_path(accwrite, omx, w, memlet=outm)

        # Rename outer connectors and add to node
        inedge._dst_conn = '_in'
        outedge._src_conn = '_out'
        node.add_in_connector('_in')
        node.add_out_connector('_out')

        from dace.transformation import dataflow
        nsdfg.apply_transformations_repeated(dataflow.MapCollapse)

        return nsdfg


@dace.library.expansion
class ExpandReduceOpenMP(pm.ExpandTransformation):
    """
        OpenMP-based implementation of the reduce node
    """
    environments = []

    _REDUCTION_TYPE_TO_OPENMP = {
        dtypes.ReductionType.Max: ('max', '{o} = max({o}, {i});'),
        dtypes.ReductionType.Min: ('min', '{o} = min({o}, {i});'),
        dtypes.ReductionType.Sum: ('+', '{o} += {i};'),
        dtypes.ReductionType.Product: ('*', '{o} *= {i};'),
        dtypes.ReductionType.Bitwise_And: ('&', '{o} &= {i};'),
        dtypes.ReductionType.Logical_And: ('&&', '{o} = {o} && {i};'),
        dtypes.ReductionType.Bitwise_Or: ('|', '{o} |= {i};'),
        dtypes.ReductionType.Logical_Or: ('||', '{o} = {o} || {i};'),
        dtypes.ReductionType.Bitwise_Xor: ('^', '{o} ^= {i};'),
        dtypes.ReductionType.Sub: ('-', '{o} -= {i};'),
        dtypes.ReductionType.Div: ('/', '{o} /= {i};'),
    }

    @staticmethod
    def expansion(node: 'Reduce', state: SDFGState, sdfg: SDFG):
        node.validate(sdfg, state)
        inedge: graph.MultiConnectorEdge = state.in_edges(node)[0]
        outedge: graph.MultiConnectorEdge = state.out_edges(node)[0]
        insubset = dcpy(inedge.data.subset)
        isqdim = insubset.squeeze()
        input_dims = len(inedge.data.subset)
        output_dims = len(outedge.data.subset)
        input_data = sdfg.arrays[inedge.data.data]
        output_data = sdfg.arrays[outedge.data.data]

        # Get reduction type for OpenMP
        redtype = detect_reduction_type(node.wcr, openmp=True)
        if redtype not in ExpandReduceOpenMP._REDUCTION_TYPE_TO_OPENMP:
            warnings.warn('Reduction type not supported for "%s"' % node.wcr)
            return ExpandReducePure.expansion(node, state, sdfg)
        omptype, expr = ExpandReduceOpenMP._REDUCTION_TYPE_TO_OPENMP[redtype]

        # Standardize axes
        axes = node.axes if node.axes is not None else [i for i in range(input_dims)]
        sqaxes = [axis for axis in axes if axis in isqdim]

        if not sqaxes:  # Degenerate reduction
            return ExpandReducePure.expansion(node, state, sdfg)

        outer_loops = len(axes) != input_dims

        # Create OpenMP clause
        if outer_loops:
            code = '#pragma omp parallel for collapse({cdim})\n'.format(cdim=output_dims)
        else:
            code = ''

        from dace.codegen.targets.cpp import sym2cpp

        # Output loops
        out_offset = []
        if outer_loops:
            for i, sz in enumerate(outedge.data.subset.size()):
                code += 'for (int _o{i} = 0; _o{i} < {sz}; ++_o{i}) {{\n'.format(i=i, sz=sym2cpp(sz))
                out_offset.append('_o%d * %s' % (i, sym2cpp(output_data.strides[i])))
        else:
            out_offset.append('0')

        outexpr = '_out[%s]' % ' + '.join(out_offset)

        # Write identity value first
        if node.identity is not None:
            code += '%s = %s;\n' % (outexpr, sym2cpp(node.identity))

        # Reduction OpenMP clause
        code += '#pragma omp parallel for collapse({cdim}) ' \
          'reduction({rtype}: {oexpr})\n'.format(cdim=len(axes), rtype=omptype,
            oexpr=outexpr)

        # Reduction loops
        for i, axis in enumerate(sorted(axes)):
            sz = sym2cpp(inedge.data.subset.size()[axis])
            code += 'for (int _i{i} = 0; _i{i} < {sz}; ++_i{i}) {{\n'.format(i=i, sz=sz)

        # Prepare input offset expression
        in_offset = []
        ictr, octr = 0, 0
        for i in range(input_dims):
            if i in axes:
                result = '_i%d' % ictr
                ictr += 1
            else:
                result = '_o%d' % octr
                octr += 1
            in_offset.append('%s * %s' % (result, sym2cpp(input_data.strides[i])))
        in_offset = ' + '.join(in_offset)

        # Reduction expression
        code += expr.format(i='_in[%s]' % in_offset, o=outexpr)
        code += '\n'

        # Closing braces
        code += '}\n' * len(axes)
        if outer_loops:
            code += '}\n' * output_dims

        # Make tasklet
        tnode = dace.nodes.Tasklet('reduce', {'_in': dace.pointer(input_data.dtype)},
                                   {'_out': dace.pointer(output_data.dtype)},
                                   code,
                                   language=dace.Language.CPP)

        # Rename outer connectors and add to node
        inedge._dst_conn = '_in'
        outedge._src_conn = '_out'
        node.add_in_connector('_in')
        node.add_out_connector('_out')

        return tnode


@dace.library.expansion
class ExpandReduceCUDADevice(pm.ExpandTransformation):
    """
        GPU implementation of the reduce node running as a device-wide kernel
        (uses CUB).
    """
    environments = [CUDA]

    _SPECIAL_RTYPES = {
        dtypes.ReductionType.Min_Location: 'ArgMin',
        dtypes.ReductionType.Max_Location: 'ArgMax',
    }

    @staticmethod
    def expansion(node: 'Reduce', state: SDFGState, sdfg: SDFG):
        from dace.codegen.prettycode import CodeIOStream
        from dace.codegen.targets.cpp import unparse_cr_split, cpp_array_expr

        node.validate(sdfg, state)
        input_edge: graph.MultiConnectorEdge = state.in_edges(node)[0]
        output_edge: graph.MultiConnectorEdge = state.out_edges(node)[0]
        insubset = dcpy(input_edge.data.subset)
        isqdim = insubset.squeeze()
        input_dims = len(input_edge.data.subset)
        output_dims = len(output_edge.data.subset)
        input_data = sdfg.arrays[input_edge.data.data]
        output_data = sdfg.arrays[output_edge.data.data]

        # Standardize axes
        axes = node.axes if node.axes is not None else [i for i in range(input_dims)]
        sqaxes = [axis for axis in axes if axis in isqdim]

        if not sqaxes:  # Degenerate reduction
            return ExpandReducePure.expansion(node, state, sdfg)

        # Setup all locations in which code will be written
        cuda_globalcode = CodeIOStream()
        cuda_initcode = CodeIOStream()
        cuda_exitcode = CodeIOStream()
        host_globalcode = CodeIOStream()
        host_localcode = CodeIOStream()
        output_memlet = output_edge.data

        # Try to autodetect reduction type
        redtype = detect_reduction_type(node.wcr)

        node_id = state.node_id(node)
        state_id = sdfg.node_id(state)
        idstr = '{sdfg}_{state}_{node}'.format(sdfg=sdfg.name, state=state_id, node=node_id)

        if node.out_connectors:
            dtype = next(node.out_connectors.values())
        else:
            dtype = sdfg.arrays[output_memlet.data].dtype

        output_type = dtype.ctype

        if node.identity is None:
            raise ValueError('For device reduce nodes, initial value must be '
                             'specified')

        # Create a functor or use an existing one for reduction
        if redtype == dtypes.ReductionType.Custom:
            body, [arg1, arg2] = unparse_cr_split(sdfg, node.wcr)
            cuda_globalcode.write(
                """
        struct __reduce_{id} {{
            template <typename T>
            DACE_HDFI T operator()(const T &{arg1}, const T &{arg2}) const {{
                {contents}
            }}
        }};""".format(id=idstr, arg1=arg1, arg2=arg2, contents=body), sdfg, state_id, node_id)
            reduce_op = ', __reduce_' + idstr + '(), ' + symstr(node.identity)
        elif redtype in ExpandReduceCUDADevice._SPECIAL_RTYPES:
            reduce_op = ''
        else:
            credtype = 'dace::ReductionType::' + str(redtype)[str(redtype).find('.') + 1:]
            reduce_op = ((', dace::_wcr_fixed<%s, %s>()' % (credtype, output_type)) + ', ' + symstr(node.identity))

        # Obtain some SDFG-related information
        input_memlet = input_edge.data
        reduce_shape = input_memlet.subset.bounding_box_size()
        num_items = ' * '.join(symstr(s) for s in reduce_shape)
        overapprox_memlet = dcpy(input_memlet)
        if any(
                str(s) not in sdfg.free_symbols.union(sdfg.constants.keys())
                for s in overapprox_memlet.subset.free_symbols):
            propagation.propagate_states(sdfg)
            for p, r in state.ranges.items():
                overapprox_memlet = propagation.propagate_subset([overapprox_memlet], input_data, [p], r)
        overapprox_shape = overapprox_memlet.subset.bounding_box_size()
        overapprox_items = ' * '.join(symstr(s) for s in overapprox_shape)

        input_dims = input_memlet.subset.dims()
        output_dims = output_memlet.subset.data_dims()

        reduce_all_axes = (node.axes is None or len(node.axes) == input_dims)
        if reduce_all_axes:
            reduce_last_axes = False
        else:
            reduce_last_axes = sorted(node.axes) == list(range(input_dims - len(node.axes), input_dims))

        if not reduce_all_axes and not reduce_last_axes:
            warnings.warn('Multiple axis reductions not supported with this expansion. '
                          'Falling back to the pure expansion.')
            return ExpandReducePureSequentialDim.expansion(node, state, sdfg)

        # Verify that data is on the GPU
        if input_data.storage not in [dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned]:
            warnings.warn('Input of GPU reduction must either reside '
                          ' in global GPU memory or pinned CPU memory')
            return ExpandReducePure.expansion(node, state, sdfg)

        if output_data.storage not in [dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned]:
            warnings.warn('Output of GPU reduction must either reside '
                          ' in global GPU memory or pinned CPU memory')
            return ExpandReducePure.expansion(node, state, sdfg)

        # Determine reduction type
        kname = (ExpandReduceCUDADevice._SPECIAL_RTYPES[redtype]
                 if redtype in ExpandReduceCUDADevice._SPECIAL_RTYPES else 'Reduce')

        # Create temp memory for this GPU
        cuda_globalcode.write(
            """
            void *__cub_storage_{sdfg}_{state}_{node} = NULL;
            size_t __cub_ssize_{sdfg}_{state}_{node} = 0;
        """.format(sdfg=sdfg.name, state=state_id, node=node_id), sdfg, state_id, node)

        if reduce_all_axes:
            reduce_type = 'DeviceReduce'
            reduce_range = overapprox_items
            reduce_range_def = 'size_t num_items'
            reduce_range_use = 'num_items'
            reduce_range_call = num_items
        elif reduce_last_axes:
            num_reduce_axes = len(node.axes)
            not_reduce_axes = reduce_shape[:-num_reduce_axes]
            reduce_axes = reduce_shape[-num_reduce_axes:]
            overapprox_not_reduce_axes = overapprox_shape[:-num_reduce_axes]
            overapprox_reduce_axes = overapprox_shape[-num_reduce_axes:]

            num_segments = ' * '.join([symstr(s) for s in not_reduce_axes])
            segment_size = ' * '.join([symstr(s) for s in reduce_axes])
            overapprox_num_segments = ' * '.join([symstr(s) for s in overapprox_not_reduce_axes])
            overapprox_segment_size = ' * '.join([symstr(s) for s in overapprox_reduce_axes])

            reduce_type = 'DeviceSegmentedReduce'
            iterator = 'dace::stridedIterator({size})'.format(size=overapprox_segment_size)
            reduce_range = '{num}, {it}, {it} + 1'.format(num=overapprox_num_segments, it=iterator)
            reduce_range_def = 'size_t num_segments, size_t segment_size'
            iterator_use = 'dace::stridedIterator(segment_size)'
            reduce_range_use = 'num_segments, {it}, {it} + 1'.format(it=iterator_use)
            reduce_range_call = '%s, %s' % (num_segments, segment_size)

        # Call CUB to get the storage size, allocate and free it
        cuda_initcode.write(
            """
            cub::{reduce_type}::{kname}(nullptr, __cub_ssize_{sdfg}_{state}_{node},
                                        ({intype}*)nullptr, ({outtype}*)nullptr, {reduce_range}{redop});
            cudaMalloc(&__cub_storage_{sdfg}_{state}_{node}, __cub_ssize_{sdfg}_{state}_{node});
""".format(sdfg=sdfg.name,
           state=state_id,
           node=node_id,
           reduce_type=reduce_type,
           reduce_range=reduce_range,
           redop=reduce_op,
           intype=input_data.dtype.ctype,
           outtype=output_data.dtype.ctype,
           kname=kname), sdfg, state_id, node)

        cuda_exitcode.write(
            'cudaFree(__cub_storage_{sdfg}_{state}_{node});'.format(sdfg=sdfg.name, state=state_id, node=node_id), sdfg,
            state_id, node)

        # Write reduction function definition
        cuda_globalcode.write("""
DACE_EXPORTED void __dace_reduce_{id}({intype} *input, {outtype} *output, {reduce_range_def}, cudaStream_t stream);
void __dace_reduce_{id}({intype} *input, {outtype} *output, {reduce_range_def}, cudaStream_t stream)
{{
cub::{reduce_type}::{kname}(__cub_storage_{id}, __cub_ssize_{id},
                            input, output, {reduce_range_use}{redop}, stream);
}}
        """.format(id=idstr,
                   intype=input_data.dtype.ctype,
                   outtype=output_data.dtype.ctype,
                   reduce_type=reduce_type,
                   reduce_range_def=reduce_range_def,
                   reduce_range_use=reduce_range_use,
                   kname=kname,
                   redop=reduce_op))

        # Write reduction function definition in caller file
        host_globalcode.write(
            """
DACE_EXPORTED void __dace_reduce_{id}({intype} *input, {outtype} *output, {reduce_range_def}, cudaStream_t stream);
        """.format(id=idstr,
                   reduce_range_def=reduce_range_def,
                   intype=input_data.dtype.ctype,
                   outtype=output_data.dtype.ctype), sdfg, state_id, node)

        # Call reduction function where necessary
        host_localcode.write('__dace_reduce_{id}(_in, _out, {reduce_range_call}, __dace_current_stream);'.format(
            id=idstr, reduce_range_call=reduce_range_call))

        # Make tasklet
        tnode = dace.nodes.Tasklet('reduce', {'_in': dace.pointer(input_data.dtype)},
                                   {'_out': dace.pointer(output_data.dtype)},
                                   host_localcode.getvalue(),
                                   language=dace.Language.CPP)

        # Add the rest of the code
        sdfg.append_global_code(host_globalcode.getvalue())
        sdfg.append_global_code(cuda_globalcode.getvalue(), 'cuda')
        sdfg.append_init_code(cuda_initcode.getvalue(), 'cuda')
        sdfg.append_exit_code(cuda_exitcode.getvalue(), 'cuda')

        # Rename outer connectors and add to node
        input_edge._dst_conn = '_in'
        output_edge._src_conn = '_out'
        node.add_in_connector('_in')
        node.add_out_connector('_out')

        return tnode


@dace.library.expansion
class ExpandReduceCUDABlock(pm.ExpandTransformation):
    """
        GPU implementation of the reduce node across a thread-block (uses CUB).
    """
    environments = [CUDA]

    _SPECIAL_RTYPES = {
        dtypes.ReductionType.Min_Location: 'ArgMin',
        dtypes.ReductionType.Max_Location: 'ArgMax',
    }

    @staticmethod
    def expansion(node: 'Reduce', state: SDFGState, sdfg: SDFG):
        from dace.codegen.prettycode import CodeIOStream
        from dace.codegen.targets.cpp import unparse_cr_split, cpp_array_expr

        node.validate(sdfg, state)
        input_edge: graph.MultiConnectorEdge = state.in_edges(node)[0]
        output_edge: graph.MultiConnectorEdge = state.out_edges(node)[0]
        input_dims = len(input_edge.data.subset)
        input_data = sdfg.arrays[input_edge.data.data]
        output_data = sdfg.arrays[output_edge.data.data]

        # Setup all locations in which code will be written
        cuda_globalcode = CodeIOStream()
        localcode = CodeIOStream()

        # Try to autodetect reduction type
        redtype = detect_reduction_type(node.wcr)

        node_id = state.node_id(node)
        state_id = sdfg.node_id(state)
        idstr = '{sdfg}_{state}_{node}'.format(sdfg=sdfg.name, state=state_id, node=node_id)

        # Obtain some SDFG-related information
        input_memlet = input_edge.data
        output_memlet = output_edge.data

        if node.out_connectors:
            dtype = next(node.out_connectors.values())
        else:
            dtype = sdfg.arrays[output_memlet.data].dtype
        output_type = dtype.ctype

        if node.identity is None:
            raise ValueError('For device reduce nodes, initial value must be '
                             'specified')

        # Create a functor or use an existing one for reduction
        if redtype == dtypes.ReductionType.Custom:
            body, [arg1, arg2] = unparse_cr_split(sdfg, node.wcr)
            cuda_globalcode.write(
                """
        struct __reduce_{id} {{
            template <typename T>
            DACE_HDFI T operator()(const T &{arg1}, const T &{arg2}) const {{
                {contents}
            }}
        }};""".format(id=idstr, arg1=arg1, arg2=arg2, contents=body), sdfg, state_id, node_id)
            reduce_op = ', __reduce_' + idstr + '(), ' + symstr(node.identity)
        elif redtype in ExpandReduceCUDADevice._SPECIAL_RTYPES:
            reduce_op = ''
        else:
            credtype = 'dace::ReductionType::' + str(redtype)[str(redtype).find('.') + 1:]
            reduce_op = ((', dace::_wcr_fixed<%s, %s>()' % (credtype, output_type)) + ', ' + symstr(node.identity))

        # Try to obtain the number of threads in the block, or use the default
        # configuration
        block_threads = devicelevel_block_size(sdfg, state, node)
        if block_threads is not None:
            block_threads = functools.reduce(lambda a, b: a * b, block_threads, 1)

        # Checks
        if block_threads is None:
            raise ValueError('Block-wide GPU reduction must occur within'
                             ' a GPU kernel')
        if issymbolic(block_threads, sdfg.constants):
            raise ValueError('Block size has to be constant for block-wide '
                             'reduction (got %s)' % str(block_threads))
        if (node.axes is not None and len(node.axes) < input_dims):
            raise ValueError('Only full reduction is supported for block-wide reduce,'
                             ' please use the pure expansion')
        if (input_data.storage != dtypes.StorageType.Register or output_data.storage != dtypes.StorageType.Register):
            raise ValueError('Block-wise reduction only supports GPU register inputs '
                             'and outputs')
        if redtype in ExpandReduceCUDABlock._SPECIAL_RTYPES:
            raise ValueError('%s block reduction not supported' % redtype)

        credtype = 'dace::ReductionType::' + str(redtype)[str(redtype).find('.') + 1:]
        if redtype == dtypes.ReductionType.Custom:
            redop = '__reduce_%s()' % idstr
        else:
            redop = 'dace::_wcr_fixed<%s, %s>()' % (credtype, output_type)

        # Allocate shared memory for block reduce
        localcode.write("""
        typedef cub::BlockReduce<{type}, {numthreads}> BlockReduce_{id};
        __shared__ typename BlockReduce_{id}::TempStorage temp_storage_{id};
            """.format(id=idstr, type=output_data.dtype.ctype, numthreads=block_threads))

        input = (input_memlet.data + ' + ' + cpp_array_expr(sdfg, input_memlet, with_brackets=False))
        output = cpp_array_expr(sdfg, output_memlet)
        localcode.write("""
            {output} = BlockReduce_{id}(temp_storage_{id}).Reduce({input}, {redop});
            """.format(id=idstr, redop=redop, input=input_memlet.data, output=output))

        # Make tasklet
        tnode = dace.nodes.Tasklet('reduce', {'_in': dace.pointer(input_data.dtype)},
                                   {'_out': dace.pointer(output_data.dtype)},
                                   localcode.getvalue(),
                                   language=dace.Language.CPP)

        # Add the rest of the code
        sdfg.append_global_code(cuda_globalcode.getvalue(), 'cuda')

        # Rename outer connectors and add to node
        input_edge._dst_conn = '_in'
        output_edge._src_conn = '_out'
        node.add_in_connector('_in')
        node.add_out_connector('_out')

        return tnode


@dace.library.expansion
class ExpandReduceCUDABlockAll(pm.ExpandTransformation):
    """ Implements the ExpandReduceCUDABlockAll transformation.
        Takes a cuda block reduce node, transforms it to a block reduce node,
         wraps it in outer maps and outputs from the root thread
        to a newly created shared memory container
    """

    environments = [CUDA]

    @staticmethod
    def redirect_edge(graph, edge, new_src=None, new_src_conn=None, new_dst=None, new_dst_conn=None, new_data=None):

        data = new_data if new_data else edge.data
        if new_src and new_dst:
            ret = graph.add_edge(new_src, new_src_conn, new_dst, new_dst_conn, data)
            graph.remove_edge(edge)
        elif new_src:
            ret = graph.add_edge(new_src, new_src_conn, edge.dst, edge.dst_conn, data)
            graph.remove_edge(edge)
        elif new_dst:
            ret = graph.add_edge(edge.src, edge.src_conn, new_dst, new_dst_conn, data)
            graph.remove_edge(edge)
        else:
            pass
        return ret

    @staticmethod
    def expansion(node: 'Reduce', state: SDFGState, sdfg: SDFG):
        """ Create a map around the BlockReduce node
            with in and out transients in registers
            and an if tasklet that redirects the output
            of thread 0 to a shared memory transient
        """
        ### define some useful vars
        graph = state
        reduce_node = node
        in_edge = graph.in_edges(reduce_node)[0]
        out_edge = graph.out_edges(reduce_node)[0]

        axes = reduce_node.axes
        ### add a map that encloses the reduce node
        (new_entry, new_exit) = graph.add_map(
                      name = 'inner_reduce_block',
                      ndrange = {'i'+str(i): f'{rng[0]}:{rng[1]+1}:{rng[2]}'  \
                                for (i,rng) in enumerate(in_edge.data.subset) \
                                if i in axes},
                      schedule = dtypes.ScheduleType.Default)

        map = new_entry.map
        ExpandReduceCUDABlockAll.redirect_edge(graph, in_edge, new_dst=new_entry)
        ExpandReduceCUDABlockAll.redirect_edge(graph, out_edge, new_src=new_exit)

        subset_in = subsets.Range([
            in_edge.data.subset[i] if i not in axes else (new_entry.map.params[0], new_entry.map.params[0], 1)
            for i in range(len(in_edge.data.subset))
        ])
        memlet_in = dace.Memlet(data=in_edge.data.data, volume=1, subset=subset_in)
        memlet_out = dcpy(out_edge.data)
        graph.add_edge(u=new_entry, u_connector=None, v=reduce_node, v_connector=None, memlet=memlet_in)
        graph.add_edge(u=reduce_node, u_connector=None, v=new_exit, v_connector=None, memlet=memlet_out)

        ### add in and out local storage
        from dace.transformation.dataflow.local_storage import LocalStorage, InLocalStorage, OutLocalStorage

        in_local_storage_subgraph = {
            LocalStorage.node_a: graph.nodes().index(new_entry),
            LocalStorage.node_b: graph.nodes().index(reduce_node)
        }
        out_local_storage_subgraph = {
            LocalStorage.node_a: graph.nodes().index(reduce_node),
            LocalStorage.node_b: graph.nodes().index(new_exit)
        }

        local_storage = InLocalStorage()
        local_storage.setup_match(sdfg, sdfg.sdfg_id, sdfg.nodes().index(state), in_local_storage_subgraph, 0)

        local_storage.array = in_edge.data.data
        local_storage.apply(graph, sdfg)
        in_transient = local_storage._data_node
        sdfg.data(in_transient.data).storage = dtypes.StorageType.Register

        local_storage = OutLocalStorage()
        local_storage.setup_match(sdfg, sdfg.sdfg_id, sdfg.nodes().index(state), out_local_storage_subgraph, 0)
        local_storage.array = out_edge.data.data
        local_storage.apply(graph, sdfg)
        out_transient = local_storage._data_node
        sdfg.data(out_transient.data).storage = dtypes.StorageType.Register

        # hack: swap edges as local_storage does not work correctly here
        # as subsets and data get assigned wrongly (should be swapped)
        # NOTE: If local_storage ever changes, this will not work any more
        e1 = graph.in_edges(out_transient)[0]
        e2 = graph.out_edges(out_transient)[0]
        e1.data.data = dcpy(e2.data.data)
        e1.data.subset = dcpy(e2.data.subset)

        ### add an if tasket and diverge
        code = 'if '
        for (i, param) in enumerate(new_entry.map.params):
            code += (param + '== 0')
            if i < len(axes) - 1:
                code += ' and '
        code += ':\n'
        code += '\tout=inp'

        tasklet_node = graph.add_tasklet(name='block_reduce_write', inputs=['inp'], outputs=['out'], code=code)

        edge_out_outtrans = graph.out_edges(out_transient)[0]
        edge_out_innerexit = graph.out_edges(new_exit)[0]
        ExpandReduceCUDABlockAll.redirect_edge(graph, edge_out_outtrans, new_dst=tasklet_node, new_dst_conn='inp')
        e = graph.add_edge(u=tasklet_node,
                           u_connector='out',
                           v=new_exit,
                           v_connector=None,
                           memlet=dcpy(edge_out_innerexit.data))
        # set dynamic with volume 0 FORNOW
        e.data.volume = 0
        e.data.dynamic = True

        ### set reduce_node axes to all (needed)
        reduce_node.axes = None

        # fill scope connectors, done.
        sdfg.fill_scope_connectors()

        # finally, change the implementation to cuda (block)
        # itself and expand again.
        reduce_node.implementation = 'CUDA (block)'
        sub_expansion = ExpandReduceCUDABlock()
        sub_expansion.setup_match(sdfg, sdfg.sdfg_id, sdfg.node_id(state), {}, 0)
        return sub_expansion.expansion(node=node, state=state, sdfg=sdfg)
        #return reduce_node.expand(sdfg, state)


@dace.library.expansion
class ExpandReduceFPGAPartialReduction(pm.ExpandTransformation):
    """
        FPGA SDFG Reduce expansion.  This does not assume single-cycle accumulation of the given data type.

        To achieve II=1, reduction is done into multiple partial reduction, which are then
        combined at the end.
    """
    environments = []

    # Reduction type expressions dictionary
    _REDUCTION_TYPE_EXPR = {
        dtypes.ReductionType.Max: 'max(prev, data_in)',
        dtypes.ReductionType.Min: 'min(prev, data_in)',
        dtypes.ReductionType.Sum: 'prev + data_in',
        dtypes.ReductionType.Product: 'prev * data_in',
        dtypes.ReductionType.Sub: 'prev - data_in',
        dtypes.ReductionType.Div: 'prev / data_in'
    }

    @staticmethod
    def expansion(node: 'Reduce', state: SDFGState, sdfg: SDFG, partial_width=16):
        '''

        :param node: the node to expand
        :param state: the state in which the node is in
        :param sdfg: the SDFG in which the node is in
        :param partial_width: Width of the inner reduction buffer. Must be
                              larger than the latency of the reduction operation on the given
                              data type
        '''
        node.validate(sdfg, state)
        inedge: graph.MultiConnectorEdge = state.in_edges(node)[0]
        outedge: graph.MultiConnectorEdge = state.out_edges(node)[0]
        input_dims = len(inedge.data.subset)
        output_dims = len(outedge.data.subset)
        input_data = sdfg.arrays[inedge.data.data]
        output_data = sdfg.arrays[outedge.data.data]

        # Standardize axes
        axes = node.axes if node.axes else [i for i in range(input_dims)]

        # Create nested SDFG
        nsdfg = SDFG('reduce')

        nsdfg.add_array('_in',
                        inedge.data.subset.size(),
                        input_data.dtype,
                        strides=input_data.strides,
                        storage=input_data.storage)

        nsdfg.add_array('_out',
                        outedge.data.subset.size(),
                        output_data.dtype,
                        strides=output_data.strides,
                        storage=output_data.storage)
        if input_data.dtype.veclen > 1:
            raise NotImplementedError('Vectorization currently not implemented for FPGA expansion of Reduce.')

        nstate = nsdfg.add_state()

        # (If axes != all) Add outer map, which corresponds to the output range
        if len(axes) != input_dims:
            all_axis = False
            # Interleave input and output axes to match input memlet
            ictr, octr = 0, 0
            input_subset = []
            for i in range(input_dims):
                if i in axes:
                    input_subset.append(f'_i{ictr}')
                    ictr += 1
                else:
                    input_subset.append(f'_o{octr}')
                    octr += 1

            output_size = outedge.data.subset.size()

            ome, omx = nstate.add_map('reduce_output',
                                      {f'_o{i}': f'0:{symstr(sz)}'
                                       for i, sz in enumerate(outedge.data.subset.size())})
            outm_idx = ','.join([f'_o{i}' for i in range(output_dims)])
            outm = dace.Memlet(f'_out[{outm_idx}]')
            inm_idx = ','.join(input_subset)
            inmm = dace.Memlet(f'_in[{inm_idx}]')
        else:
            all_axis = True
            ome, omx = None, None
            outm = dace.Memlet('_out[0]')
            inm_idx = ','.join([f'_i{i}' for i in range(len(axes))])
            inmm = dace.Memlet(f'_in[{inm_idx}]')

        # Add inner map, which corresponds to the range to reduce
        r = nstate.add_read('_in')
        w = nstate.add_read('_out')

        # TODO support vectorization
        buffer_name = 'partial_results'
        nsdfg.add_array(buffer_name, (partial_width, ),
                        input_data.dtype,
                        transient=True,
                        storage=dtypes.StorageType.FPGA_Local)
        buffer = nstate.add_access(buffer_name)
        buffer_write = nstate.add_write(buffer_name)

        # Initialize explicitly partial results, as the inner map could run for a number of iteration < partial_width
        init_me, init_mx = nstate.add_map('partial_results_init', {'i': f'0:{partial_width}'},
                                          schedule=dtypes.ScheduleType.FPGA_Device,
                                          unroll=True)
        init_tasklet = nstate.add_tasklet('init_pr', {}, {'pr_out'}, f'pr_out = {node.identity}')
        nstate.add_memlet_path(init_me, init_tasklet, memlet=dace.Memlet())
        nstate.add_memlet_path(init_tasklet,
                               init_mx,
                               buffer,
                               src_conn='pr_out',
                               memlet=dace.Memlet(f'{buffer_name}[i]'))

        if not all_axis:
            nstate.add_memlet_path(ome, init_me, memlet=dace.Memlet())

        ime, imx = nstate.add_map(
            'reduce_values',
            {f'_i{i}': f'0:{symstr(inedge.data.subset.size()[axis])}'
             for i, axis in enumerate(sorted(axes))})

        # Accumulate over partial results
        redtype = detect_reduction_type(node.wcr)
        if redtype not in ExpandReduceFPGAPartialReduction._REDUCTION_TYPE_EXPR:
            raise ValueError('Reduction type not supported for "%s"' % node.wcr)
        else:
            reduction_expr = ExpandReduceFPGAPartialReduction._REDUCTION_TYPE_EXPR[redtype]

        # generate flatten index considering inner map: will be used for indexing into partial results
        ranges_size = ime.range.size()
        inner_index = '+'.join([f'_i{i} * {ranges_size[i + 1]}' for i in range(len(axes) - 1)])
        inner_op = ' + ' if len(axes) > 1 else ''
        inner_index = inner_index + f'{inner_op}_i{(len(axes) - 1)}'
        partial_reduce_tasklet = nstate.add_tasklet('partial_reduce', {'data_in', 'buffer_in'}, {'buffer_out'}, f'''\
prev = buffer_in
buffer_out = {reduction_expr}''')

        if not all_axis:
            # Connect input and partial sums
            nstate.add_memlet_path(r, ome, ime, partial_reduce_tasklet, dst_conn='data_in', memlet=inmm)
        else:
            nstate.add_memlet_path(r, ime, partial_reduce_tasklet, dst_conn='data_in', memlet=inmm)
        nstate.add_memlet_path(buffer,
                               ime,
                               partial_reduce_tasklet,
                               dst_conn='buffer_in',
                               memlet=dace.Memlet(f'{buffer_name}[({inner_index})%{partial_width}]'))
        nstate.add_memlet_path(partial_reduce_tasklet,
                               imx,
                               buffer_write,
                               src_conn='buffer_out',
                               memlet=dace.Memlet(f'{buffer_name}[({inner_index})%{partial_width}]'))

        # Then perform reduction on partial results
        reduce_entry, reduce_exit = nstate.add_map('reduce', {'i': f'0:{partial_width}'},
                                                   schedule=dtypes.ScheduleType.FPGA_Device,
                                                   unroll=True)

        reduce_tasklet = nstate.add_tasklet(
            'reduce', {'reduce_in', 'data_in'}, {'reduce_out'}, f'''\
prev = reduce_in if i > 0 else {node.identity}
reduce_out = {reduction_expr}''')
        nstate.add_memlet_path(buffer_write,
                               reduce_entry,
                               reduce_tasklet,
                               dst_conn='data_in',
                               memlet=dace.Memlet(f'{buffer_name}[i]'))

        reduce_name = 'reduce_result'
        nsdfg.add_array(reduce_name, (1, ), output_data.dtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        reduce_read = nstate.add_access(reduce_name)
        reduce_access = nstate.add_access(reduce_name)

        if not all_axis:
            nstate.add_memlet_path(ome, reduce_read, memlet=dace.Memlet())

        nstate.add_memlet_path(reduce_read,
                               reduce_entry,
                               reduce_tasklet,
                               dst_conn='reduce_in',
                               memlet=dace.Memlet(f'{reduce_name}[0]'))
        nstate.add_memlet_path(reduce_tasklet,
                               reduce_exit,
                               reduce_access,
                               src_conn='reduce_out',
                               memlet=dace.Memlet(f'{reduce_name}[0]'))

        if not all_axis:
            # Write out the result
            nstate.add_memlet_path(reduce_access, omx, w, memlet=outm)
        else:
            nstate.add_memlet_path(reduce_access, w, memlet=outm)

        # Rename outer connectors and add to node
        inedge._dst_conn = '_in'
        outedge._src_conn = '_out'
        node.add_in_connector('_in')
        node.add_out_connector('_out')
        nsdfg.validate()

        return nsdfg


@dace.library.expansion
class WarpReductionExpansion(pm.ExpandTransformation):
    
    
    
    environments = [CUDA]
    
    @staticmethod
    def expansion(node: 'Reduce', parent_state: SDFGState, parent_sdfg: SDFG, local_sum: SDFG.node, local_Result: SDFG.node):
        
        # ToDo: fix the data descriptors!
            # The data we recieve:
                # FROM PARENT'S PARENT
                # warpSize      this should be a free symbol for now I am just using a hardcoded 32 to see if warpSize was the thing causing the segfault
                # N
                
                
        # ToDo: implement checks (once this thing works when used appropriately)
        
        ###################
        # Grabbing the Data:
        
        print("Entered my expansion")
        
        input_edge = parent_state.in_edges(node)[0]
        output_edge = parent_state.out_edges(node)[0]
        
        input_data = parent_sdfg.arrays[input_edge.data]      # this I believe to be the location of sA, sB and mySum (the data) for now, this can probably be fixed by the way the library node is called (since sA and sB aren't needed)
        output_data = parent_sdfg.arrays[output_edge.data]     # and this the location of sRes
        
        # in_ptr = dace.pointer(input_data.dtype)
        # out_ptr = dace.pointer(output_data.dtype)
        
        
        ###################
        
        ###################
        # Notes to self:
        # 
        # Can I access an access node of the parent sdfg (gpu_sdfg) in this state?
        
        ###################
        
        def WriteBackState(state, mS, r):
    
            src_node = state.add_read(mS)
            dst_node = state.add_write(r)
            
            state.add_nedge(src_node, dst_node, dace.Memlet(f"{r}", wcr="lambda x, y: x + y"))      

        
        subSDFG = dace.SDFG('WarpReduction_SDFG')
        
        ###################
        # Adding the GPU map
        
        gpuCallState = subSDFG.add_state()

        me, mx = gpuCallState.add_map('GPU_map', {'i': '0:min(int_ceil(N, BlockDim), GridDim)', 'j': '0:BlockDim'})
        me.map.schedule = dace.dtypes.ScheduleType.GPU_Device
        
        ###################
        # Creating the SDFG with the actual Reduction Loop
        
        gpu_sdfg = dace.SDFG('GPU_SDFG')
        
        # How do we pass on Symbols?
        # For now I have hardcoded the value for N
        gpu_sdfg.add_array('sA', shape= [10000], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
        gpu_sdfg.add_array('sRes', shape=[1], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
        gpu_sdfg.add_scalar('mySum', dtype=dace.float64, storage=dace.StorageType.Register, transient=True)

        
        wwr_state= gpu_sdfg.add_state('WarpWise_Reduction')

        mSum = wwr_state.add_access(local_sum)
        wtasklet = wwr_state.add_tasklet('warpwise_Reduction',
                                        {}, {'__out'},
                                        f"__out = __shfl_down_sync(0xFFFFFFFF, {mSum.data}, offset);",
                                        dace.Language.CPP)
        wwr_state.add_edge(wtasklet, '__out', local_sum, None, dace.Memlet(f"{local_sum.data}", wcr="lambda x, y: x + y"))

        _, _, after_state = gpu_sdfg.add_loop(parent_state, wwr_state, None, 'offset', '32 / 2', 'offset > 0', 'offset / 2')

        write_back_state = gpu_sdfg.add_state('Write_Back')
        WriteBackState(write_back_state, local_sum, local_Result)

        random_end_state = gpu_sdfg.add_state('RandomEndState')

        gpu_sdfg.add_edge(after_state, write_back_state, dace.InterstateEdge('j % 32 == 0'))
        gpu_sdfg.add_edge(after_state, random_end_state, dace.InterstateEdge('j % 32 != 0'))
        gpu_sdfg.add_edge(write_back_state, random_end_state, dace.InterstateEdge())
        
        ###################
        # Make the dataflow between the states happen
        da_whole_SDFG = gpuCallState.add_nested_sdfg(gpu_sdfg, subSDFG, {'sA'}, {'sRes'})

        Ain = gpuCallState.add_read(input_data)
        ROut = gpuCallState.add_write(output_data)

        gpuCallState.add_memlet_path(Ain, me, da_whole_SDFG, memlet=dace.Memlet(data=input_data, subset='0: (min(int_ceil(N, BlockDim), GridDim) * BlockDim)'), dst_conn='sA')
        gpuCallState.add_memlet_path(da_whole_SDFG, mx, ROut, memlet=dace.Memlet(data=output_data, subset='0'), src_conn='sRes')


        subSDFG.apply_transformations_repeated(pm.MapExpansion)

        for n in gpuCallState.nodes():
            if isinstance(n, nodes.MapEntry) and "j" in n.map.params:
                n.map.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock
       
        return subSDFG





@dace.library.node
class Reduce(dace.sdfg.nodes.LibraryNode):
    """ An SDFG node that reduces an N-dimensional array to an
        (N-k)-dimensional array, with a list of axes to reduce and
        a reduction binary function. """

    # Global properties
    implementations = {
        'pure': ExpandReducePure,
        'pure-seq': ExpandReducePureSequentialDim,
        'OpenMP': ExpandReduceOpenMP,
        'CUDA (device)': ExpandReduceCUDADevice,
        'CUDA (block)': ExpandReduceCUDABlock,
        'CUDA (block allreduce)': ExpandReduceCUDABlockAll,
        'FPGAPartialReduction': ExpandReduceFPGAPartialReduction,
        'CUDA(shuffle)' : WarpReductionExpansion
        # 'CUDA (warp)': ExpandReduceCUDAWarp,
        # 'CUDA (warp allreduce)': ExpandReduceCUDAWarpAll
    }

    default_implementation = 'CUDA(shuffle)'

    # Properties
    axes = ListProperty(element_type=int, allow_none=True)
    wcr = LambdaProperty(default='lambda a, b: a')
    identity = Property(allow_none=True)

    def __init__(self,
                 wcr='lambda a, b: a',
                 axes=None,
                 identity=None,
                 schedule=dtypes.ScheduleType.Default,
                 debuginfo=None,
                 **kwargs):
        super().__init__(name='Reduce', **kwargs)
        self.wcr = wcr
        self.axes = axes
        self.identity = identity
        self.debuginfo = debuginfo
        self.schedule = schedule

    @staticmethod
    def from_json(json_obj, context=None):
        ret = Reduce("lambda a, b: a", None)
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    def __str__(self):
        # Autodetect reduction type
        redtype = detect_reduction_type(self.wcr)
        if redtype == dtypes.ReductionType.Custom:
            wcrstr = unparse(ast.parse(self.wcr).body[0].value.body)
        else:
            wcrstr = str(redtype)
            wcrstr = wcrstr[wcrstr.find('.') + 1:]  # Skip "ReductionType."

        return 'Reduce ({op}), Axes: {axes}'.format(axes=('all' if self.axes is None else str(self.axes)), op=wcrstr)

    def __label__(self, sdfg, state):
        return str(self).replace(' Axes', '\nAxes')

    def validate(self, sdfg, state):
        if len(state.in_edges(self)) != 1:
            raise ValueError('Reduce node must have one input')
        if len(state.out_edges(self)) != 1:
            raise ValueError('Reduce node must have one output')
