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
from dace.libraries.standard.environments.cuda import CUDA
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.cpp import unparse_cr_split, cpp_array_expr


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

        nsdfg.add_array('_out',
                        outedge.data.subset.size(),
                        output_data.dtype,
                        strides=output_data.strides,
                        storage=output_data.storage)

        # If identity is defined, add an initialization state
        if node.identity is not None:
            init_state = nsdfg.add_state()
            nstate = nsdfg.add_state()
            nsdfg.add_edge(init_state, nstate, dace.InterstateEdge())

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
            nstate = nsdfg.add_state()
        # END OF INIT

        # (If axes != all) Add outer map, which corresponds to the output range
        if len(axes) != input_dims:
            # Interleave input and output axes to match input memlet
            ictr, octr = 0, 0
            input_subset = []
            for i in range(input_dims):
                if i in axes:
                    input_subset.append('_i%d' % ictr)
                    ictr += 1
                else:
                    input_subset.append('_o%d' % octr)
                    octr += 1

            output_size = outedge.data.subset.size()

            ome, omx = nstate.add_map(
                'reduce_output', {
                    '_o%d' % i: '0:%s' % symstr(sz)
                    for i, sz in enumerate(outedge.data.subset.size())
                })
            outm = dace.Memlet.simple(
                '_out',
                ','.join(['_o%d' % i for i in range(output_dims)]),
                wcr_str=node.wcr)
            inmm = dace.Memlet.simple('_in', ','.join(input_subset))
        else:
            ome, omx = None, None
            outm = dace.Memlet.simple('_out', '0', wcr_str=node.wcr)
            inmm = dace.Memlet.simple(
                '_in', ','.join(['_i%d' % i for i in range(len(axes))]))

        # Add inner map, which corresponds to the range to reduce, containing
        # an identity tasklet
        ime, imx = nstate.add_map(
            'reduce_values', {
                '_i%d' % i: '0:%s' % symstr(inedge.data.subset.size()[axis])
                for i, axis in enumerate(sorted(axes))
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
        input_dims = len(inedge.data.subset)
        output_dims = len(outedge.data.subset)
        input_data = sdfg.arrays[inedge.data.data]
        output_data = sdfg.arrays[outedge.data.data]

        # Get reduction type for OpenMP
        redtype = detect_reduction_type(node.wcr, openmp=True)
        if redtype not in ExpandReduceOpenMP._REDUCTION_TYPE_TO_OPENMP:
            raise ValueError('Reduction type not supported for "%s"' %
                             node.wcr)
        omptype, expr = ExpandReduceOpenMP._REDUCTION_TYPE_TO_OPENMP[redtype]

        # Standardize axes
        axes = node.axes if node.axes else [i for i in range(input_dims)]

        outer_loops = len(axes) != input_dims

        # Create OpenMP clause
        if outer_loops:
            code = '#pragma omp parallel for collapse({cdim})\n'.format(
                cdim=output_dims)
        else:
            code = ''

        from dace.codegen.targets.cpp import sym2cpp

        # Output loops
        out_offset = []
        if outer_loops:
            for i, sz in enumerate(outedge.data.subset.size()):
                code += 'for (int _o{i} = 0; _o{i} < {sz}; ++_o{i}) {{\n'.format(
                    i=i, sz=sym2cpp(sz))
                out_offset.append('_o%d * %s' %
                                  (i, sym2cpp(output_data.strides[i])))

        # HACK: This works around the current codegen, which infers scalars
        if outedge.data.subset.num_elements() == 1:
            outexpr = '_out'
        else:
            outexpr = '_out[%s]' % ' + '.join(out_offset)
        # END HACK

        # Write identity value first
        if node.identity is not None:
            code += '%s = %s;\n' % (outexpr, node.identity)

        # Reduction OpenMP clause
        code += '#pragma omp parallel for collapse({cdim}) ' \
          'reduction({rtype}: {oexpr})\n'.format(cdim=len(axes), rtype=omptype,
            oexpr=outexpr)

        # Reduction loops
        for i, axis in enumerate(sorted(axes)):
            sz = sym2cpp(inedge.data.subset.size()[axis])
            code += 'for (int _i{i} = 0; _i{i} < {sz}; ++_i{i}) {{\n'.format(
                i=i, sz=sz)

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
            in_offset.append('%s * %s' %
                             (result, sym2cpp(input_data.strides[i])))
        in_offset = ' + '.join(in_offset)

        # Reduction expression
        code += expr.format(i='_in[%s]' % in_offset, o=outexpr)
        code += '\n'

        # Closing braces
        code += '}\n' * len(axes)
        if outer_loops:
            code += '}\n' * output_dims

        # Make tasklet
        tnode = dace.nodes.Tasklet('reduce', {'_in'}, {'_out'},
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
        node.validate(sdfg, state)
        input_edge: graph.MultiConnectorEdge = state.in_edges(node)[0]
        output_edge: graph.MultiConnectorEdge = state.out_edges(node)[0]
        input_dims = len(input_edge.data.subset)
        output_dims = len(output_edge.data.subset)
        input_data = sdfg.arrays[input_edge.data.data]
        output_data = sdfg.arrays[output_edge.data.data]

        # Setup all locations in which code will be written
        cuda_globalcode = CodeIOStream()
        cuda_initcode = CodeIOStream()
        cuda_exitcode = CodeIOStream()
        host_globalcode = CodeIOStream()
        host_localcode = CodeIOStream()

        # Try to autodetect reduction type
        redtype = detect_reduction_type(node.wcr)

        node_id = state.node_id(node)
        state_id = sdfg.node_id(state)
        idstr = '{sdfg}_{state}_{node}'.format(sdfg=sdfg.name,
                                               state=state_id,
                                               node=node_id)

        output_memlet = output_edge.data
        output_type = 'dace::vec<%s, %s>' % (
            sdfg.arrays[output_memlet.data].dtype.ctype, output_memlet.veclen)

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
        }};""".format(id=idstr, arg1=arg1, arg2=arg2, contents=body), sdfg,
                state_id, node_id)
            reduce_op = ', __reduce_' + idstr + '(), ' + symstr(node.identity)
        elif redtype in ExpandReduceCUDADevice._SPECIAL_RTYPES:
            reduce_op = ''
        else:
            credtype = 'dace::ReductionType::' + str(
                redtype)[str(redtype).find('.') + 1:]
            reduce_op = ((', dace::_wcr_fixed<%s, %s>()' %
                          (credtype, output_type)) + ', ' +
                         symstr(node.identity))

        # Obtain some SDFG-related information
        input_access = state.memlet_path(input_edge)[0].src
        output_access = state.memlet_path(output_edge)[-1].dst
        input_memlet = input_edge.data
        reduce_shape = input_memlet.subset.bounding_box_size()
        num_items = ' * '.join(symstr(s) for s in reduce_shape)
        input = (input_memlet.data + ' + ' +
                 cpp_array_expr(sdfg, input_memlet, with_brackets=False))
        output = (output_memlet.data + ' + ' +
                  cpp_array_expr(sdfg, output_memlet, with_brackets=False))

        input_dims = input_memlet.subset.dims()
        output_dims = output_memlet.subset.data_dims()

        reduce_all_axes = (node.axes is None or len(node.axes) == input_dims)
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
        if input_data.storage not in [
                dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned
        ]:
            raise ValueError('Input of GPU reduction must either reside '
                             ' in global GPU memory or pinned CPU memory')
        if output_data.storage not in [
                dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned
        ]:
            raise ValueError('Output of GPU reduction must either reside '
                             ' in global GPU memory or pinned CPU memory')

        # Determine reduction type
        kname = (ExpandReduceCUDADevice._SPECIAL_RTYPES[redtype]
                 if redtype in ExpandReduceCUDADevice._SPECIAL_RTYPES else
                 'Reduce')

        # Create temp memory for this GPU
        cuda_globalcode.write(
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

            num_segments = ' * '.join([symstr(s) for s in not_reduce_axes])
            segment_size = ' * '.join([symstr(s) for s in reduce_axes])

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
            'cudaFree(__cub_storage_{sdfg}_{state}_{node});'.format(
                sdfg=sdfg.name, state=state_id, node=node_id), sdfg, state_id,
            node)

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
        host_localcode.write(
            '__dace_reduce_{id}({input}, {output}, {reduce_range_call}, __dace_current_stream);'
            .format(id=idstr,
                    input=input,
                    output=output,
                    reduce_range_call=reduce_range_call))

        # Make tasklet
        tnode = dace.nodes.Tasklet('reduce', {'_in'}, {'_out'},
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

        # HACK: Workaround to avoid issues with code generator inferring reads
        # and writes when it shouldn't.
        input_edge.data.num_accesses = dtypes.DYNAMIC
        output_edge.data.num_accesses = dtypes.DYNAMIC

        return tnode


# TODO: Convert to ExpandReduceCUDA/CUB/...
'''
def _generate_Reduce(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        

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
            self.cuda_initcode.write(
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

'''


@dace.library.node
class Reduce(dace.graph.nodes.LibraryNode):
    """ An SDFG node that reduces an N-dimensional array to an
        (N-k)-dimensional array, with a list of axes to reduce and
        a reduction binary function. """

    # Global properties
    implementations = {
        'pure': ExpandReducePure,
        'OpenMP': ExpandReduceOpenMP,
        'CUDA (device)': ExpandReduceCUDADevice,
        # 'CUDA (block)': ExpandReduceCUDABlock,
        # 'CUDA (warp)': ExpandReduceCUDAWarp,
        # 'CUDA (warp allreduce)': ExpandReduceCUDAWarpAllreduce
    }

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

        return 'Reduce ({op}), Axes: {axes}'.format(
            axes=('all' if self.axes is None else str(self.axes)), op=wcrstr)

    def __label__(self, sdfg, state):
        return str(self).replace(' Axes', '\nAxes')

    def validate(self, sdfg, state):
        if len(state.in_edges(self)) != 1:
            raise ValueError('Reduce node must have one input')
        if len(state.out_edges(self)) != 1:
            raise ValueError('Reduce node must have one output')
