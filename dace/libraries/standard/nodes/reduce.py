""" File defining the reduction library node. """

import ast
from copy import deepcopy as dcpy
import dace
import itertools
import dace.serialize
import dace.library
from typing import Any, Dict, Set
from dace.config import Config
from dace.sdfg import SDFG, SDFGState
from dace.graph import graph
from dace.frontend.python.astutils import unparse
from dace.properties import (Property, CodeProperty, LambdaProperty,
                             RangeProperty, DebugInfoProperty, SetProperty,
                             make_properties, indirect_properties,
                             DataProperty, SymbolicProperty, ListProperty,
                             SDFGReferenceProperty, DictProperty,
                             LibraryImplementationProperty)
from dace.frontend.operations import detect_reduction_type
from dace import data, subsets as sbs, dtypes
from dace import registry, subsets
import pydoc
import warnings
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching as pm
from dace.symbolic import symstr


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
        output_strides = None
        if len(axes) != input_dims:
            output_strides = [
                s for i, s in enumerate(output_data.strides) if i not in axes
            ]
        nsdfg.add_array('_out',
                        outedge.data.subset.size(),
                        output_data.dtype,
                        strides=output_strides,
                        storage=output_data.storage)

        # If identity is defined, add an initialization state
        if node.identity is not None:
            init_state = nsdfg.add_state()
            nstate = nsdfg.add_state()
            nsdfg.add_edge(init_state, nstate, dace.InterstateEdge())

            if len(axes) != input_dims:
                # Add initialization as a map
                init_state.add_mapped_tasklet(
                    'reduce_init', {
                        '_o%d' % i: '0:%s' % symstr(d)
                        for i, d in enumerate(outedge.data.subset.size())
                    }, {},
                    'out = %s' % node.identity, {
                        'out':
                        dace.Memlet.simple(
                            '_out', ','.join(
                                ['_o%d' % i for i in range(output_dims)]))
                    },
                    external_edges=True)
            else:
                # Add initialization as a tasklet
                t = init_state.add_tasklet('reduce_init', {}, {'out'},
                                           'out = %s' % node.identity)
                w = init_state.add_write('_out')
                init_state.add_edge(t, 'out', w, None,
                                    dace.Memlet.simple('_out', '0'))
        else:
            nstate = nsdfg.add_state()
        # END OF INIT

        # (If axes != all) Add outer map, which corresponds to the output range
        if len(axes) != input_dims:
            output_axes = [i for i in range(input_dims) if i not in axes]
            output_size = outedge.data.subset.size()

            ome, omx = nstate.add_map(
                'reduce_output', {
                    '_o%d' % i: '0:%s' % symstr(sz)
                    for i, sz in zip(output_axes, output_size)
                })
            outm = dace.Memlet.simple('_out',
                                      ','.join(
                                          ['_o%d' % i for i in output_axes]),
                                      wcr_str=node.wcr)
            inmm = dace.Memlet.simple(
                '_in', ','.join([
                    '_i%d' % i if i in axes else '_o%d' % i
                    for i in range(input_dims)
                ]))
        else:
            ome, omx = None, None
            outm = dace.Memlet.simple('_out', '0', wcr_str=node.wcr)
            inmm = dace.Memlet.simple(
                '_in', ','.join(['_i%d' % i for i in range(len(axes))]))

        # Add inner map, which corresponds to the range to reduce, containing
        # an identity tasklet
        ime, imx = nstate.add_map(
            'reduce_values', {
                '_i%d' % axis: '0:%s' % symstr(inedge.data.subset.size()[axis])
                for axis in axes
            })

        # Add identity tasklet for reduction
        t = nstate.add_tasklet('identity', {'inp'}, {'out'}, 'out = inp')

        # Connect everything
        r = nstate.add_read('_in')
        w = nstate.add_read('_out')
        if ome:
            nstate.add_memlet_path(r, ome, ime, t, dst_conn='inp', memlet=inmm)
            nstate.add_memlet_path(t, imx, omx, w, src_conn='out', memlet=outm)
        else:
            nstate.add_memlet_path(r, ime, t, dst_conn='inp', memlet=inmm)
            nstate.add_memlet_path(t, imx, w, src_conn='out', memlet=outm)

        # Rename outer connectors and add to node
        inedge._dst_conn = '_in'
        outedge._src_conn = '_out'
        node.add_in_connector('_in')
        node.add_out_connector('_out')

        return nsdfg


@dace.library.node
class Reduce(dace.graph.nodes.LibraryNode):
    """ An SDFG node that reduces an N-dimensional array to an
        (N-k)-dimensional array, with a list of axes to reduce and
        a reduction binary function. """

    # Global properties
    implementations = {
        'pure': ExpandReducePure,
    }  # 'OpenMP': ExpandReduceOpenMP,
    # 'CUDA (atomic): ExpandReduceCUDA,
    # 'CUDA (shared)': ExpandReduceCUDABlock,
    # 'CUDA (warp-level intrinsics)': ExpandReduceCUDAWarp
    default_implementation = 'pure'

    # Properties
    axes = ListProperty(element_type=int, allow_none=True)
    wcr = LambdaProperty(default='lambda a, b: a')
    identity = Property(allow_none=True)
    debuginfo = DebugInfoProperty()

    def __init__(self,
                 wcr='lambda a, b: a',
                 axes=None,
                 identity=None,
                 debuginfo=None,
                 **kwargs):
        super().__init__(name='Reduce', **kwargs)
        self.wcr = wcr
        self.axes = axes
        self.identity = identity
        self.debuginfo = debuginfo

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

        return 'Reduce ({op}), Axes: {axes}'.format(
            axes=('all' if self.axes is None else str(self.axes)), op=wcrstr)

    def __label__(self, sdfg, state):
        return str(self).replace(' Axes', '\nAxes')

    def validate(self, sdfg, state):
        if len(state.in_edges(self)) != 1:
            raise ValueError('Reduce node must have one input')
        if len(state.out_edges(self)) != 1:
            raise ValueError('Reduce node must have one output')


# TODO: Convert to ExpandReduceOpenMP
'''
    def _generate_Reduce(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        # Try to autodetect reduction type
        redtype = operations.detect_reduction_type(node.wcr)

        loop_header = ""

        if node.schedule == dtypes.ScheduleType.CPU_Multicore:
            loop_header += "#pragma omp parallel for"

        end_braces = 0

        axes = node.axes
        state_dfg = sdfg.nodes()[state_id]
        input_memlet = state_dfg.in_edges(node)[0].data
        output_edge = state_dfg.out_edges(node)[0]
        output_memlet = output_edge.data

        output_type = "dace::vec<%s, %s>" % (
            sdfg.arrays[output_memlet.data].dtype.ctype,
            output_memlet.veclen,
        )

        # Example:
        #
        # inp = array(K, M, N)
        # out = array(L, M, N)
        # out[i] = reduce(lambda a, b: a + b, inp, axes=0)
        #
        # Pseudocode:
        # for m in range(M):
        #     for n in range(N):
        #         for k in range(K):
        #             out[i, m, n] += inp[k, m, n]

        # The number of output dimensions is equal to the number of input
        # dimensions, minus the number of dimensions being reduced (axes)
        # Example:
        # 3D array inp reduced to a 2D array
        # NOTE: The number of output dimensions is less or equal to the number
        # of dimensions of the output array, where the result is stored.
        input_num_dims = input_memlet.subset.dims()
        # If axes were not defined, use all input dimensions
        if axes is None:
            axes = tuple(range(input_num_dims))
        output_num_dims = input_num_dims - len(axes)

        # Obtain variable names per output and reduction axis
        axis_vars = []  # Iteration variables for the input dimensions
        output_axis_vars = dict()  # Dict matching the dimensions of the input
        # that are NOT being reduced with the
        # equivalent dimensions of the output array.
        output_dims = []  # The equivalent output array dimensions
        # of the input dimensions NOT being reduced.
        octr = 0  # First index for the dimensions NOT being reduced.
        input_size = input_memlet.subset.size()
        output_size = output_memlet.subset.size()
        for d in range(input_num_dims):
            if d in axes:
                # Dimension is being reduced.
                axis_vars.append(symbolic.pystr_to_symbolic("__i%d" % d))
            elif input_size[d] != 1:
                # Dimension is NOT being reduced.
                axis_vars.append(symbolic.pystr_to_symbolic("__o%d" % octr))
                ri = input_size[d]
                # Iterate over the dimensions of the output memlet and find
                # the one that is matching with the input memlet dimension.
                for i, ro in enumerate(output_size):
                    # This is needed in case where there are multiple NOT
                    # reduced dimensions with the same size.
                    if i in output_axis_vars.keys():
                        continue
                    if ri == ro:
                        output_dims.append(i)
                        output_axis_vars[i] = octr
                        break
                octr += 1
            else:  # input_size == 1, no additional offset necessary
                axis_vars.append(0)
        # Example:
        # __i0 -> first dimension of inp (size K)
        # __o0 -> second dimension of inp and out (size M)
        # __o1 -> third dimensions of inp and out (size N)

        # Instrumentation: Post-scope
        instr = self._dispatcher.instrumentation[node.instrument]
        inner_stream = None
        if instr is not None:
            inner_stream = CodeIOStream()
            instr.on_node_begin(sdfg, state_dfg, node, callsite_stream,
                                inner_stream, function_stream)

        # Write OpenMP loop pragma if there are output dimensions
        if output_num_dims > 0:
            callsite_stream.write(loop_header, sdfg, state_id, node)

        # Generate outer loops
        output_subset = output_memlet.subset
        for axis in output_dims:
            octr = output_axis_vars[axis]
            callsite_stream.write(
                "for (int {var} = {begin}; {var} < {end}; ++{var}) {{".format(
                    var="__o%d" % octr,
                    begin=0,
                    end=output_size[axis],
                ),
                sdfg,
                state_id,
                node,
            )

            end_braces += 1
        # Example:
        # for (int __o0 = 0; __o0 < M; __o0 += 1) {
        #     for (int __o1 = 0; __o1 < N; __o1 += 1) {

        use_tmpout = False
        if len(axes) == input_num_dims:
            # Add OpenMP reduction clause if reducing all axes
            if (redtype != dtypes.ReductionType.Custom
                    and node.schedule == dtypes.ScheduleType.CPU_Multicore):
                loop_header += " reduction(%s: __tmpout)" % (
                    _REDUCTION_TYPE_TO_OPENMP[redtype])

            # Output initialization
            identity = ""
            if node.identity is not None:
                identity = " = %s" % sym2cpp(node.identity)
            callsite_stream.write(
                "{\n%s __tmpout%s;" % (output_type, identity), sdfg, state_id,
                node)
            callsite_stream.write(loop_header, sdfg, state_id, node)
            end_braces += 1
            use_tmpout = True

        # Compute output expression
        outvar = ("__tmpout" if use_tmpout else cpp_array_expr(
            sdfg,
            output_memlet,
            indices=[
                symbolic.pystr_to_symbolic("__o%d" % output_axis_vars[i])
                if i in output_axis_vars.keys() else 0
                for i, r in enumerate(output_subset)
            ],
        ))
        # Example (pseudocode):
        # out[i, __o0, __o1]

        if len(axes) != input_num_dims and node.identity is not None:
            # Write code for identity value in multiple axes
            callsite_stream.write(
                "%s = %s;" % (outvar, sym2cpp(node.identity)), sdfg, state_id,
                node)

        # Instrumentation: internal part
        if instr is not None:
            callsite_stream.write(inner_stream.getvalue())

        # Generate inner loops (reducing)
        input_subset = input_memlet.subset
        for axis in axes:
            callsite_stream.write(
                "for (int {var} = {begin}; {var} < {end}; ++{var}) {{".format(
                    var="__i%d" % axis,
                    begin=0,
                    end=input_size[axis],
                ),
                sdfg,
                state_id,
                node,
            )
            end_braces += 1
        # Example:
        # for (int __i0 = 0; __i0 < K; __i0 += 1) {

        # Generate reduction code
        credtype = "dace::ReductionType::" + str(
            redtype)[str(redtype).find(".") + 1:]

        atomic_suffix = ('_atomic' if output_memlet.wcr_conflict else '')

        invar = cpp_array_expr(sdfg, input_memlet, indices=axis_vars)

        if redtype != dtypes.ReductionType.Custom:
            callsite_stream.write(
                "dace::wcr_fixed<%s, %s>::reduce%s(&%s, %s);" %
                (credtype, output_type, atomic_suffix, outvar, invar),
                sdfg,
                state_id,
                node,
            )  # cpp_array_expr(), cpp_array_expr()
        else:
            callsite_stream.write(
                'dace::wcr_custom<%s>::template reduce%s(%s, &%s, %s);' %
                (output_type, atomic_suffix, unparse_cr(
                    sdfg, node.wcr), outvar, invar), sdfg, state_id,
                node)  #cpp_array_expr(), cpp_array_expr()

        #############################################################
        # Generate closing braces
        outer_stream = None
        for i in range(end_braces):
            if i == len(axes):
                # Instrumentation: post-scope
                if instr is not None:
                    outer_stream = CodeIOStream()
                    instr.on_node_end(sdfg, state_dfg, node, outer_stream,
                                      callsite_stream, function_stream)

            # Store back tmpout into the true output
            if i == end_braces - 1 and use_tmpout:
                if (self._dispatcher.defined_vars.get(
                        output_memlet.data) == DefinedType.Scalar):
                    out_var = output_memlet.data
                else:
                    out_var = cpp_array_expr(sdfg, output_memlet)
                callsite_stream.write(
                    "%s = __tmpout;" % out_var,
                    sdfg,
                    state_id,
                    node,
                )

            callsite_stream.write("}", sdfg, state_id, node)

        if instr is not None:
            callsite_stream.write(outer_stream.getvalue())
'''

# TODO: Convert to ExpandReduceCUDA/CUB/...
'''
def _generate_Reduce(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        # Try to autodetect reduction type
        redtype = operations.detect_reduction_type(node.wcr)
        schedule = node.schedule
        node_id = dfg.node_id(node)
        idstr = '{sdfg}_{state}_{node}'.format(sdfg=sdfg.name,
                                               state=state_id,
                                               node=node_id)

        output_edge = dfg.out_edges(node)[0]
        output_memlet = output_edge.data
        output_type = 'dace::vec<%s, %s>' % (
            sdfg.arrays[output_memlet.data].dtype.ctype, output_memlet.veclen)

        if node.identity is None:
            raise ValueError('For GPU reduce nodes, initial value must be '
                             'defined')

        # Create a functor or use an existing one for reduction
        if redtype == dtypes.ReductionType.Custom:
            body, [arg1, arg2] = unparse_cr_split(sdfg, node.wcr)
            self._globalcode.write(
                """
        struct __reduce_{id} {{
            template <typename T>
            DACE_HDFI T operator()(const T &{arg1}, const T &{arg2}) const {{
                {contents}
            }}
        }};""".format(id=idstr, arg1=arg1, arg2=arg2, contents=body), sdfg,
                state_id, node_id)
            reduce_op = ', __reduce_' + idstr + '(), ' + _topy(node.identity)
        elif redtype in _SPECIAL_RTYPES:
            reduce_op = ''
        else:
            credtype = 'dace::ReductionType::' + str(
                redtype)[str(redtype).find('.') + 1:]
            reduce_op = ((', dace::_wcr_fixed<%s, %s>()' %
                          (credtype, output_type)) + ', ' +
                         _topy(node.identity))

        # Obtain some SDFG-related information
        input_data = dfg.memlet_path(dfg.in_edges(node)[0])[0].src
        output_data = dfg.memlet_path(dfg.out_edges(node)[0])[-1].dst
        input_memlet = dfg.in_edges(node)[0].data
        reduce_shape = input_memlet.subset.bounding_box_size()
        num_items = ' * '.join([_topy(s) for s in reduce_shape])
        input = (input_memlet.data + ' + ' +
                 cpp_array_expr(sdfg, input_memlet, with_brackets=False))
        output = (output_memlet.data + ' + ' +
                  cpp_array_expr(sdfg, output_memlet, with_brackets=False))

        # Options: Device-wide reduction (even from device code),
        #          block-wide reduction, sequential reduction (for loop)
        if node.schedule == dtypes.ScheduleType.GPU_Device:

            input_dims = input_memlet.subset.dims()
            output_dims = output_memlet.subset.data_dims()

            reduce_all_axes = (node.axes is None
                               or len(node.axes) == input_dims)
            if reduce_all_axes:
                reduce_last_axes = False
            else:
                reduce_last_axes = sorted(node.axes) == list(
                    range(input_dims - len(node.axes), input_dims))

            if (not reduce_all_axes) and (not reduce_last_axes):
                raise NotImplementedError(
                    'Multiple axis reductions not supported on GPUs. Please '
                    'apply ReduceExpansion or make reduce axes to be last in the array'
                )

            # Verify that data is on the GPU
            if input_data.desc(sdfg).storage not in [
                    dtypes.StorageType.GPU_Global,
                    dtypes.StorageType.CPU_Pinned
            ]:
                raise ValueError('Input of GPU reduction must either reside '
                                 ' in global GPU memory or pinned CPU memory')
            if output_data.desc(sdfg).storage not in [
                    dtypes.StorageType.GPU_Global,
                    dtypes.StorageType.CPU_Pinned
            ]:
                raise ValueError('Output of GPU reduction must either reside '
                                 ' in global GPU memory or pinned CPU memory')

            # TODO(later): Enable device-wide reduction from device through
            # CUDA dynamic parallelism. It is disabled right now
            # due to temporary memory allocation (which needs to be done
            # on the host).
            if self._in_device_code:
                raise NotImplementedError('Device-wide reduction can only be'
                                          ' run on non-GPU code.')

            # Determine reduction type
            kname = (_SPECIAL_RTYPES[redtype]
                     if redtype in _SPECIAL_RTYPES else 'Reduce')

            # Create temp memory for this GPU
            self._globalcode.write(
                """
                void *__cub_storage_{sdfg}_{state}_{node} = NULL;
                size_t __cub_ssize_{sdfg}_{state}_{node} = 0;
            """.format(sdfg=sdfg.name, state=state_id, node=node_id), sdfg,
                state_id, node)

            if reduce_all_axes:
                reduce_type = 'DeviceReduce'
                reduce_range = num_items
                reduce_range_def = 'size_t num_items'
                reduce_range_use = 'num_items'
                reduce_range_call = num_items
            elif reduce_last_axes:
                num_reduce_axes = len(node.axes)
                not_reduce_axes = reduce_shape[:-num_reduce_axes]
                reduce_axes = reduce_shape[-num_reduce_axes:]

                num_segments = ' * '.join([_topy(s) for s in not_reduce_axes])
                segment_size = ' * '.join([_topy(s) for s in reduce_axes])

                reduce_type = 'DeviceSegmentedReduce'
                iterator = 'dace::stridedIterator({size})'.format(
                    size=segment_size)
                reduce_range = '{num}, {it}, {it} + 1'.format(num=num_segments,
                                                              it=iterator)
                reduce_range_def = 'size_t num_segments, size_t segment_size'
                iterator_use = 'dace::stridedIterator(segment_size)'
                reduce_range_use = 'num_segments, {it}, {it} + 1'.format(
                    it=iterator_use)
                reduce_range_call = '%s, %s' % (num_segments, segment_size)

            # Call CUB to get the storage size, allocate and free it
            self.scope_entry_stream.write(
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
            intype=input_data.desc(sdfg).dtype.ctype,
            outtype=output_data.desc(sdfg).dtype.ctype,
            kname=kname), sdfg, state_id, node)

            self.scope_exit_stream.write(
                'cudaFree(__cub_storage_{sdfg}_{state}_{node});'.format(
                    sdfg=sdfg.name, state=state_id, node=node_id), sdfg,
                state_id, node)

            max_streams = int(
                Config.get('compiler', 'cuda', 'max_concurrent_streams'))
            if max_streams >= 0:
                cudastream = 'dace::cuda::__streams[%d]' % node._cuda_stream
            else:
                cudastream = 'nullptr'

            # Write reduction function definition
            self._localcode.write(
                """
DACE_EXPORTED void __dace_reduce_{id}({intype} *input, {outtype} *output, {reduce_range_def});
void __dace_reduce_{id}({intype} *input, {outtype} *output, {reduce_range_def})
{{
    cub::{reduce_type}::{kname}(__cub_storage_{id}, __cub_ssize_{id},
                                input, output, {reduce_range_use}{redop}, {stream});
}}
            """.format(id=idstr,
                       intype=input_data.desc(sdfg).dtype.ctype,
                       outtype=output_data.desc(sdfg).dtype.ctype,
                       reduce_type=reduce_type,
                       reduce_range_def=reduce_range_def,
                       reduce_range_use=reduce_range_use,
                       kname=kname,
                       redop=reduce_op,
                       stream=cudastream), sdfg, state_id, node)

            # Write reduction function definition in caller file
            function_stream.write(
                """
DACE_EXPORTED void __dace_reduce_{id}({intype} *input, {outtype} *output, {reduce_range_def});
            """.format(id=idstr,
                       reduce_range_def=reduce_range_def,
                       intype=input_data.desc(sdfg).dtype.ctype,
                       outtype=output_data.desc(sdfg).dtype.ctype), sdfg,
                state_id, node)

            # Call reduction function where necessary
            callsite_stream.write(
                '__dace_reduce_{id}({input}, {output}, {reduce_range_call});'.
                format(id=idstr,
                       input=input,
                       output=output,
                       reduce_range_call=reduce_range_call), sdfg, state_id,
                node)

            synchronize_streams(sdfg, dfg, state_id, node, node,
                                callsite_stream)
            return

        # Block-wide reduction
        elif node.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
            input_dims = input_memlet.subset.dims()
            # Checks
            if not self._in_device_code:
                raise ValueError('Block-wide GPU reduction must occur within'
                                 ' a GPU kernel')
            for bdim in self._block_dims:
                if symbolic.issymbolic(bdim, sdfg.constants):
                    raise ValueError(
                        'Block size has to be constant for block-wide '
                        'reduction (got %s)' % str(bdim))
            if (node.axes is not None and len(node.axes) < input_dims):
                raise ValueError(
                    'Only full reduction is supported for block-wide reduce,'
                    ' please use ReduceExpansion')
            if (input_data.desc(sdfg).storage != dtypes.StorageType.GPU_Stack
                    or output_data.desc(sdfg).storage !=
                    dtypes.StorageType.GPU_Stack):
                raise ValueError(
                    'Block-wise reduction only supports GPU register inputs '
                    'and outputs')
            if redtype in _SPECIAL_RTYPES:
                raise ValueError('%s block reduction not supported' % redtype)

            credtype = 'dace::ReductionType::' + str(
                redtype)[str(redtype).find('.') + 1:]
            if redtype == dtypes.ReductionType.Custom:
                redop = '__reduce_%s()' % idstr
            else:
                redop = 'dace::_wcr_fixed<%s, %s>()' % (credtype, output_type)

            # Allocate shared memory for block reduce
            self.scope_entry_stream.write(
                """
            typedef cub::BlockReduce<{type}, {numthreads}> BlockReduce_{id};
            __shared__ typename BlockReduce_{id}::TempStorage temp_storage_{id};
                """.format(id=idstr,
                           type=output_data.desc(sdfg).dtype.ctype,
                           numthreads=' * '.join(
                               str(s) for s in self._block_dims)), sdfg,
                state_id, node)

            # TODO(later): If less than the whole block is participating,
            #              use special CUB function
            output = cpp_array_expr(sdfg, output_memlet)
            callsite_stream.write(
                """
                {output} = BlockReduce_{id}(temp_storage_{id}).Reduce({input}, {redop});
                """.format(id=idstr,
                           redop=redop,
                           input=input_memlet.data,
                           output=output), sdfg, state_id, node)

            return
        # Sequential goes to CPU generator
        elif node.schedule == dtypes.ScheduleType.Sequential:
            self._cpu_codegen._generate_Reduce(sdfg, dfg, state_id, node,
                                               function_stream,
                                               callsite_stream)
            return
        else:
            raise ValueError('Unsupported reduction schedule %s' %
                             str(node.schedule))
'''