# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" File defining the reduction library node. """

import ast
from copy import deepcopy as dcpy
import dace
import functools
import numpy
import platform
import dace.serialize
import dace.library
from dace.sdfg import SDFG, SDFGState, devicelevel_block_size, propagation
from dace.sdfg import graph
from dace.frontend.python.astutils import unparse
from dace.properties import Property, LambdaProperty, ListProperty
from dace.frontend.operations import detect_reduction_type
from dace import dtypes
from dace import subsets
import warnings
from dace.sdfg import scope
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.transformation import transformation as pm
from dace.symbolic import symstr, issymbolic, simplify
from dace.libraries.standard.environments.cuda import CUDA

from dace.libraries.standard import reduction_planner as red_planner

#: Output storage locations a GPU device reduction (a CUB ``DeviceReduce`` launch or a device-map
#: reduction schedule) can write into directly. ``GPU_Global`` is device memory; ``CPU_Pinned`` is
#: page-locked host memory that is device-addressable. Any other storage
#: (``Register`` / ``CPU_Heap`` / ``Default`` / ...) is host-only: a device reduction must then write
#: a ``GPU_Global`` scratch and copy the result out at the storage boundary (see
#: :func:`route_gpu_reduce_result_to_host_output`).
GPU_REDUCE_DEVICE_WRITABLE_STORAGE = (dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned)


def route_gpu_reduce_result_to_host_output(nsdfg: SDFG, last_state: SDFGState, dtype, out_shape, out_strides,
                                           host_storage: dtypes.StorageType) -> None:
    """Rewire a GPU-reduction nested SDFG whose result was written into a device array named ``_out``
    so that ``_out`` becomes the real (host) output connector, fed by a device->host copy.

    The reduction body built ``_out`` as ``GPU_Global`` device memory (a device kernel cannot write
    host memory). This renames that device buffer to a ``GPU_Global`` transient scratch, adds a fresh
    non-transient ``_out`` in the real output's (host) storage, and appends a state copying the
    scratch to it -- the device->host copy DaCe emits at the storage boundary. Value-exact: the whole
    reduced result is copied, and any WCR the reduction applied is already folded into the scratch.

    :param nsdfg: the reduction nested SDFG whose current ``_out`` array holds the device result.
    :param last_state: the terminal state that wrote the device result (the copy is appended after it).
    :param dtype: the output element type.
    :param out_shape: the reduced-output shape (the scratch and host ``_out`` share it).
    :param out_strides: the host output's strides.
    :param host_storage: the real output's (host) storage type.
    """
    nsdfg.replace('_out', '_out_gpu')
    scratch = nsdfg.arrays['_out_gpu']
    scratch.transient = True
    scratch.storage = dtypes.StorageType.GPU_Global
    nsdfg.add_array('_out', out_shape, dtype, strides=out_strides, storage=host_storage)
    copy_state = nsdfg.add_state_after(last_state)
    copy_state.add_nedge(copy_state.add_read('_out_gpu'), copy_state.add_write('_out'),
                         dace.Memlet.from_array('_out', nsdfg.arrays['_out']))


#: Namespaced tasklet connector names: after nested-SDFG inlining a bare ``a``/``b``/``o`` would
#: collide with the ``AllNode``/``AnyNode`` ``_out`` array and fail validation.
_IN = '_reduce_in'
_OUT = '_reduce_out'
_ACC = '_reduce_acc'

#: ``Reduce``'s data connectors. Module-level as in ``copy/common.py`` and ``scan``, so an expansion
#: or a caller can name them without importing the node class.
INPUT_CONNECTOR_NAME = '_in'
OUTPUT_CONNECTOR_NAME = '_out'


@dace.library.expansion
class ExpandReducePure(pm.ExpandTransformation):
    """
        Pure SDFG Reduce expansion replaces a reduce node with nested maps and
        edges with WCR.
    """
    environments = []

    @staticmethod
    def map_schedules(node: 'Reduce', state: SDFGState, sdfg: SDFG):
        """``(outermost, inner)`` schedule for the maps this expansion builds.

        Schedule inference has already run by the time a library node expands, so a map left at the
        default schedule is whatever the code around it happens to be -- and on device-resident data
        at host level that is a host loop over ``GPU_Global`` memory (npbench nbody, where the GPU
        expansion declines a node carrying no identity and lands here). Inside a kernel the opposite
        holds: everything below is device code already and a nested device map is not allowed.
        """
        default = (dtypes.ScheduleType.Default, dtypes.ScheduleType.Default)
        if scope.is_devicelevel_gpu(sdfg, state, node):
            return dtypes.ScheduleType.Sequential, dtypes.ScheduleType.Sequential
        operands = [state.in_edges(node)[0].data.data, state.out_edges(node)[0].data.data]
        if any(sdfg.arrays[name].storage in GPU_RESIDENT_STORAGES for name in operands):
            return dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.Sequential
        return default

    @staticmethod
    def expansion(node: 'Reduce', state: SDFGState, sdfg: SDFG):
        node.validate(sdfg, state)
        outer_schedule, inner_schedule = ExpandReducePure.map_schedules(node, state, sdfg)
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

        # The maps below are named ``_o<n>`` / ``_i<n>``. When a symbol of that name is already defined
        # where this node sits -- a reduce nested inside a map over ``_o0``, which is what
        # ReduceExpansion's out-transient path builds -- the inner map rebinds it, and the boundary
        # memlet naming the OUTER symbol then reads the inner map's value. That is silent: every output
        # element comes out as the first one. Suffix the names until they are free at this scope.
        taken = {str(s) for s in state.symbols_defined_at(node).keys()}
        suffix = ''
        while any('_o%d%s' % (i, suffix) in taken or '_i%d%s' % (i, suffix) in taken
                  for i in range(max(input_dims, output_dims) + 1)):
            suffix += '_'

        def oname(i: int) -> str:
            return '_o%d%s' % (i, suffix)

        def iname(i: int) -> str:
            return '_i%d%s' % (i, suffix)

        # Create nested SDFG
        nsdfg = SDFG('reduce')

        # Kept-dim stride = ARRAY stride × input subset STEP → strided input (a[0:2N:2])
        # indexes right elements; subset begin folded into _in ptr by caller. Step-1 unchanged.
        nsdfg.add_array('_in',
                        insubset.size(),
                        input_data.dtype,
                        strides=[input_data.strides[orig] * insubset[j][2] for j, orig in enumerate(isqdim)],
                        storage=input_data.storage)

        nsdfg.add_array('_out',
                        outsubset.size(),
                        output_data.dtype,
                        strides=[s for i, s in enumerate(output_data.strides) if i in osqdim],
                        storage=output_data.storage)

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
            init_state.add_mapped_tasklet('reduce_init', {
                oname(i): '0:%s' % symstr(d)
                for i, d in enumerate(outedge.data.subset.size())
            }, {},
                                          '__out = %s' % node.identity,
                                          {'__out': dace.Memlet.simple('_out', ','.join([oname(i) for i in osqdim]))},
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
                    input_subset.append(iname(ictr))
                    ictr += 1
                else:
                    input_subset.append(oname(octr))
                    octr += 1

            ome, omx = nstate.add_map('reduce_output', {
                oname(i): '0:%s' % symstr(sz)
                for i, sz in enumerate(outsubset.size())
            },
                                      schedule=outer_schedule)
            outm = dace.Memlet.simple('_out', ','.join([oname(i) for i in range(output_dims)]), wcr_str=node.wcr)
            inmm = dace.Memlet.simple('_in', ','.join(input_subset))
        else:
            ome, omx = None, None
            outm = dace.Memlet.simple('_out', '0', wcr_str=node.wcr)
            inmm = dace.Memlet.simple('_in', ','.join([iname(i) for i in range(len(axes))]))

        # Add inner map, which corresponds to the range to reduce, containing
        # an identity tasklet
        # With no outer map the inner one IS the outermost scope, so it carries that schedule.
        ime, imx = nstate.add_map('reduce_values', {
            iname(i): '0:%s' % symstr(insubset.size()[isqdim.index(axis)])
            for i, axis in enumerate(sorted(axes))
        },
                                  schedule=inner_schedule if ome is not None else outer_schedule)

        # Add identity tasklet for reduction
        t = nstate.add_tasklet('identity', {'__inp': None}, {'__out': None}, '__out = __inp')

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
        # ``validate=False``: this SDFG is still DETACHED, so nothing here can see the scope it is
        # about to be placed in. A device-resident operand read by the Sequential maps above then
        # reads as a host access of GPU memory -- which it is not, because the node these maps
        # replace is inside a kernel. The graph is validated in context by the caller.
        nsdfg.apply_transformations_repeated(dataflow.MapCollapse, validate=False)

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

        # Same naming hazard as in ExpandReducePure: a map named after a symbol already defined at this
        # scope rebinds it, and the boundary memlet naming the outer one then reads the inner map's
        # value -- silently, with every output element equal to the first.
        taken = {str(sym) for sym in state.symbols_defined_at(node).keys()}
        suffix = ''
        while any('_o%d%s' % (i, suffix) in taken or '_i%d%s' % (i, suffix) in taken
                  for i in range(max(input_dims, output_dims) + 1)):
            suffix += '_'

        def oname(i: int) -> str:
            return '_o%d%s' % (i, suffix)

        def iname(i: int) -> str:
            return '_i%d%s' % (i, suffix)

        # Create nested SDFG
        nsdfg = SDFG('reduce')

        # Kept-dim stride = ARRAY stride × input subset STEP → strided input (a[0:2N:2])
        # indexes right elements; subset begin folded into _in ptr by caller. Step-1 unchanged.
        nsdfg.add_array('_in',
                        insubset.size(),
                        input_data.dtype,
                        strides=[input_data.strides[orig] * insubset[j][2] for j, orig in enumerate(isqdim)],
                        storage=input_data.storage)

        nsdfg.add_array('_out',
                        outsubset.size(),
                        output_data.dtype,
                        strides=[s for i, s in enumerate(output_data.strides) if i in osqdim],
                        storage=output_data.storage)

        # The accumulator is the OUTPUT element type. That is THE CONTRACT -- ``dace::reduce::sum<T, U>``
        # names ``T`` the seed/output type and ``U`` the input's and casts per element, and
        # ``reduce_scan_dtype_matrix_test.test_reduce_accumulates_in_the_output_dtype`` pins it. Of the
        # three host expansions this was the only one that broke it: ``ExpandReduceOpenMP`` goes
        # through that runtime entry point and ``ExpandReducePure`` writes its WCR straight into
        # ``_out``, while this one STAGES an accumulator and used to stage it at the INPUT's type. It
        # is also the one ``ExpandReduceAuto`` dispatches to for every ``Sequential`` reduction that
        # carries an identity, so an int8 predicate mask summed into an int64 total wrapped at 127.
        nsdfg.add_transient('acc', [1], nsdfg.arrays['_out'].dtype, dtypes.StorageType.Register)

        nstate = nsdfg.add_state()

        # Interleave input and output axes to match input memlet
        ictr, octr = 0, 0
        input_subset = []
        for i in isqdim:
            if i in axes:
                input_subset.append(iname(ictr))
                ictr += 1
            else:
                input_subset.append(oname(octr))
                octr += 1

        ome, omx = nstate.add_map('reduce_output', {
            oname(i): '0:%s' % symstr(sz)
            for i, sz in enumerate(outsubset.size())
        })
        outm = dace.Memlet.simple('_out', ','.join([oname(i) for i in range(output_dims)]))
        #wcr_str=node.wcr)
        inmm = dace.Memlet.simple('_in', ','.join(input_subset))

        idt = nstate.add_tasklet('reset', {}, {_OUT}, f'{_OUT} = {node.identity}')
        nstate.add_edge(ome, None, idt, None, dace.Memlet())

        accread = nstate.add_access('acc')
        accwrite = nstate.add_access('acc')
        nstate.add_edge(idt, _OUT, accread, None, dace.Memlet('acc'))

        # Add inner map, which corresponds to the range to reduce, containing
        # an identity tasklet
        ime, imx = nstate.add_map('reduce_values', {
            iname(i): '0:%s' % symstr(insubset.size()[isqdim.index(axis)])
            for i, axis in enumerate(sorted(axes))
        },
                                  schedule=dtypes.ScheduleType.Sequential)

        # Add identity tasklet for reduction
        t = nstate.add_tasklet('identity', {_ACC, _IN}, {_OUT}, f'{_OUT} = {_IN}')

        # Connect everything
        r = nstate.add_read('_in')
        w = nstate.add_write('_out')
        nstate.add_memlet_path(r, ome, ime, t, dst_conn=_IN, memlet=inmm)
        nstate.add_memlet_path(accread, ime, t, dst_conn=_ACC, memlet=dace.Memlet('acc[0]'))
        nstate.add_memlet_path(t, imx, accwrite, src_conn=_OUT, memlet=dace.Memlet('acc[0]', wcr=node.wcr))
        # Same dtype by construction now, so the store is a plain copy edge; the cast tasklet the
        # input-typed accumulator needed is gone with it. The widening happens per element instead,
        # on the identity tasklet's connectors, which is where it belongs.
        nstate.add_memlet_path(accwrite, omx, w, memlet=outm)

        from dace.transformation import dataflow
        # ``validate=False``: this SDFG is still DETACHED, so nothing here can see the scope it is
        # about to be placed in. A device-resident operand read by the Sequential maps above then
        # reads as a host access of GPU memory -- which it is not, because the node these maps
        # replace is inside a kernel. The graph is validated in context by the caller.
        nsdfg.apply_transformations_repeated(dataflow.MapCollapse, validate=False)

        return nsdfg


def stage_gpu_reduction_output(node: 'Reduce', state: SDFGState, sdfg: SDFG):
    """Route a GPU reduction whose destination is not device-resident through a device transient.

    The device expansions write through a device pointer, so a host destination -- typically a
    scalar reduced over every axis -- reaches codegen as an illegal copy. Reduce into a one-element
    GPU_Global transient instead and let the edge out of it lower to the device-to-host copy.
    """
    outedge = state.out_edges(node)[0]
    desc = sdfg.arrays[outedge.data.data]
    if desc.storage in (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared):
        return

    name, _ = sdfg.add_array(f'{node.label}_gpu_out', [1],
                             desc.dtype,
                             storage=dtypes.StorageType.GPU_Global,
                             transient=True,
                             find_new_name=True)
    staged = state.add_access(name)
    state.add_edge(node, outedge.src_conn, staged, None, dace.Memlet(f'{name}[0:1]'))
    state.add_edge(staged, None, outedge.dst, outedge.dst_conn, dcpy(outedge.data))
    state.remove_edge(outedge)


@dace.library.expansion
class ExpandReduceAuto(pm.ExpandTransformation):
    """
        Dispatches to one of the existing expansions based on the node's schedule, which
        ``set_default_schedule_and_storage_types`` assigns before expansion:

        * ``Sequential`` (the node is nested in a parallel map) -> the sequential accumulator,
          whose combination order is fixed and therefore independent of the thread count. It needs
          an identity to seed the accumulator, so a node without one goes the parallel way.
        * a GPU schedule -> ``ExpandReduceGPUAuto``, which plans the device schedule itself and
          falls back to the pure expansion when it cannot.
        * ``Default``, i.e. nobody inferred one -> the pure map+WCR expansion. This is not codegen
          (``codegen.py`` infers schedules *before* it expands), it is a caller running
          ``expand_library_nodes()`` on a graph it still intends to transform, and the OpenMP
          expansion returns an opaque C++ tasklet that no dataflow transformation can see into
          (``LiftEinsum`` stops recognising the reduction, fusion stops fusing it).
        * any other concrete schedule -> OpenMP, which emits a real ``reduction()`` clause instead
          of a per-element atomic. It reassociates, so it is not reproducible across thread counts;
          that is the accepted cost of a parallel reduction.
    """
    environments = []

    @staticmethod
    def expansion(node: 'Reduce', state: SDFGState, sdfg: SDFG):
        ExpandReduceAuto.environments = []
        if node.schedule == dtypes.ScheduleType.Sequential and node.identity is not None:
            return ExpandReducePureSequentialDim.expansion(node, state, sdfg)
        if node.schedule in dtypes.GPU_SCHEDULES:
            stage_gpu_reduction_output(node, state, sdfg)
            expanded = ExpandReduceGPUAuto.expansion(node, state, sdfg)
            # The GPU expansion picks its own environments when it delegates to CUB
            ExpandReduceAuto.environments = list(ExpandReduceGPUAuto.environments)
            return expanded
        if node.schedule == dtypes.ScheduleType.Default:
            # Un-inferred: stay in dataflow form so later passes can still match the reduction.
            return ExpandReducePure.expansion(node, state, sdfg)
        return ExpandReduceOpenMP.expansion(node, state, sdfg)


@dace.library.expansion
class ExpandReduceOpenMP(pm.ExpandTransformation):
    """CPU lowering of the ``Reduce`` node onto the ``dace::reduce`` runtime facility.

    One call per output element into ``dace/reduction.h``: ``::dace::reduce::<op>`` only when this
    node IS the outermost parallel work, ``::dace::reduce::seq::<op>`` under an enclosing
    ``omp parallel for`` over the kept axes or any enclosing parallel map / loop -- the entries have
    no runtime nesting check, so SCOPE decides that here, statically. The ``reduction`` clause is the
    header's -- no pragma, no per-element atomic here. SHAPES: the reduced axes split into the
    innermost CONTIGUOUS run (consecutive indices whose strides nest exactly, hence one
    ``count``/``stride`` walk) plus the rest; the run is the call, the rest are plain C loops around
    it, and no strided input is ever packed, so a single-axis reduction (the ``LoopToReduce`` shape)
    and a full dense reduction are each ONE call with no loops. A degenerate reduction, unrecognised
    WCR, non-scalar dtype or Windows falls back to :class:`ExpandReducePure`.
    """
    environments = []

    #: ReductionType -> (``dace::reduce`` entry point, seed expression, statement applying the call).
    #: An associative op seeds from the output element and assigns back, so the ``identity`` write is
    #: folded in by the call and an empty range is a no-op. ``Sub`` is ``initial - sum(x)`` (what
    #: OpenMP's deprecated ``-`` does), ``Div`` is ``initial / prod(x)``, the only parallel ``/``.
    _REDUCTION_TYPE_TO_RUNTIME = {
        dtypes.ReductionType.Max: ('max', '{o}', '{o} = {call};'),
        dtypes.ReductionType.Min: ('min', '{o}', '{o} = {call};'),
        dtypes.ReductionType.Sum: ('sum', '{o}', '{o} = {call};'),
        dtypes.ReductionType.Product: ('product', '{o}', '{o} = {call};'),
        dtypes.ReductionType.Bitwise_And: ('bitwise_and', '{o}', '{o} = {call};'),
        dtypes.ReductionType.Logical_And: ('logical_and', '{o}', '{o} = {call};'),
        dtypes.ReductionType.Bitwise_Or: ('bitwise_or', '{o}', '{o} = {call};'),
        dtypes.ReductionType.Logical_Or: ('logical_or', '{o}', '{o} = {call};'),
        dtypes.ReductionType.Bitwise_Xor: ('bitwise_xor', '{o}', '{o} = {call};'),
        dtypes.ReductionType.Sub: ('sum', '({t})0', '{o} -= {call};'),
        dtypes.ReductionType.Div: ('product', '({t})1', '{o} = {o} / {call};'),
    }

    #: Integer-only: no ``&`` / ``|`` / ``^`` exists on a floating-point operand, in C++ or in OpenMP.
    BITWISE_REDUCTIONS = (
        dtypes.ReductionType.Bitwise_And,
        dtypes.ReductionType.Bitwise_Or,
        dtypes.ReductionType.Bitwise_Xor,
    )

    #: Real-only: a complex operand has neither an ordering nor a boolean conversion, so C++ has no
    #: ``<`` / ``&&`` for it and OpenMP no ``min`` / ``max`` reduction. ``+`` and ``*`` are the only
    #: reductions defined on complex -- the same policy ``dace::reduce::detail::real_seed`` asserts.
    REAL_ONLY_REDUCTIONS = (
        dtypes.ReductionType.Min,
        dtypes.ReductionType.Max,
        dtypes.ReductionType.Logical_And,
        dtypes.ReductionType.Logical_Or,
    )

    @staticmethod
    def contiguous_tail(raxes, sizes, strides):
        """Longest suffix of ``raxes`` that is one contiguous walk: consecutive indices whose strides
        nest exactly (``stride[a] == stride[a + 1] * size[a + 1]``), so the run is addressable as
        ``count`` elements ``stride`` apart. An unprovable symbolic pair ends it (more loops)."""
        cut = len(raxes) - 1
        while cut > 0:
            outer, inner = raxes[cut - 1], raxes[cut]
            if inner != outer + 1:
                break
            try:
                if bool(simplify(strides[outer] - strides[inner] * sizes[inner]) != 0):
                    break
            except TypeError:
                break
            cut -= 1
        return raxes[cut:]

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

        if platform.system() == 'Windows':
            warnings.warn('OpenMP reduction expansion not supported on Visual C++')
            return ExpandReducePure.expansion(node, state, sdfg)

        # Scalar element types only; a struct / vector / pointer dtype has no OpenMP reduction.
        if type(input_data.dtype) is not dtypes.typeclass or type(output_data.dtype) is not dtypes.typeclass:
            warnings.warn('Reduction element type %s -> %s is not a scalar type, using the pure expansion' %
                          (input_data.dtype, output_data.dtype))
            return ExpandReducePure.expansion(node, state, sdfg)

        redtype = detect_reduction_type(node.wcr, openmp=True)

        if redtype in ExpandReduceOpenMP.BITWISE_REDUCTIONS:
            elemtype = output_data.dtype
            if not numpy.issubdtype(elemtype.type, numpy.integer) and elemtype != dtypes.bool_:
                raise ValueError('Bitwise reduction "%s" is not defined for non-integral data type %s '
                                 '(reducing into "%s"). Bitwise operators do not exist for floating-point '
                                 'or complex operands in C++, and OpenMP has no reduction for them either. '
                                 'Use a logical reduction (&&, ||) or an integer dtype.' %
                                 (node.wcr, elemtype, outedge.data.data))

        if redtype in ExpandReduceOpenMP.REAL_ONLY_REDUCTIONS:
            for elemtype, name in ((input_data.dtype, inedge.data.data), (output_data.dtype, outedge.data.data)):
                if numpy.issubdtype(elemtype.type, numpy.complexfloating):
                    raise ValueError('Reduction "%s" is not defined for complex data type %s (on "%s"). '
                                     'A complex operand has no ordering and no boolean conversion, so C++ has '
                                     'no "<" / "&&" for it and OpenMP no min/max reduction. Only sum and '
                                     'product are defined on complex.' % (node.wcr, elemtype, name))

        if redtype not in ExpandReduceOpenMP._REDUCTION_TYPE_TO_RUNTIME:
            warnings.warn('Reduction type not supported for "%s"' % node.wcr)
            return ExpandReducePure.expansion(node, state, sdfg)
        fname, seed_fmt, stmt = ExpandReduceOpenMP._REDUCTION_TYPE_TO_RUNTIME[redtype]

        axes = node.axes if node.axes is not None else [i for i in range(input_dims)]
        sqaxes = [axis for axis in axes if axis in isqdim]

        if not sqaxes:  # Degenerate reduction
            return ExpandReducePure.expansion(node, state, sdfg)

        from dace.codegen.targets.cpp import sym2cpp

        raxes = sorted(axes)
        sizes = {a: inedge.data.subset.size()[a] for a in raxes}
        strides = {a: input_data.strides[a] * inedge.data.subset[a][2] for a in raxes}
        group = ExpandReduceOpenMP.contiguous_tail(raxes, sizes, strides)
        outer_red = raxes[:len(raxes) - len(group)]

        outer_loops = len(axes) != input_dims

        # SCOPE decides the shape, not ``node.schedule``: that is storage-derived, so a Reduce nested
        # in a parallel map arrives carrying ``CPU_Multicore``. A re-entered node opens no region.
        from dace.transformation.auto.auto_optimize import libnode_is_sequential
        reentered = libnode_is_sequential(node, state, sdfg)
        parallel_outputs = outer_loops and not reentered

        def collapse_clause(ndims):
            return 'collapse(%d) ' % ndims if ndims > 1 else ''

        # Output loops carry a partial reduction's parallelism; ``collapse(1)`` is the no-op default.
        code = ''
        out_offset = []
        if outer_loops:
            if parallel_outputs:
                code += ('#pragma omp parallel for ' + collapse_clause(output_dims)).rstrip() + '\n'
            for i, sz in enumerate(outedge.data.subset.size()):
                code += 'for (int _o{i} = 0; _o{i} < {sz}; ++_o{i}) {{\n'.format(i=i, sz=sym2cpp(sz))
                out_offset.append('_o%d * %s' % (i, sym2cpp(output_data.strides[i])))
        else:
            out_offset.append('0')
        outexpr = '%s[%s]' % (_OUT, ' + '.join(out_offset))

        if node.identity is not None:
            code += '%s = %s;\n' % (outexpr, sym2cpp(node.identity))

        for i, axis in enumerate(outer_red):
            code += 'for (int _i{i} = 0; _i{i} < {sz}; ++_i{i}) {{\n'.format(i=i, sz=sym2cpp(sizes[axis]))

        in_offset = []
        octr = 0
        for i in range(input_dims):
            stride = sym2cpp(input_data.strides[i] * inedge.data.subset[i][2])
            if i in axes:
                if i in outer_red:
                    in_offset.append('_i%d * %s' % (outer_red.index(i), stride))
            else:
                in_offset.append('_o%d * %s' % (octr, stride))
                octr += 1

        count = functools.reduce(lambda a, b: a * b, [sizes[a] for a in group], 1)
        call = '::dace::reduce::{seq}{fn}({base}, (long)({count}), (long)({stride}), {seed})'.format(
            seq='seq::' if (outer_loops or reentered) else '',
            fn=fname,
            base=_IN if not in_offset else '%s + %s' % (_IN, ' + '.join(in_offset)),
            count=sym2cpp(count),
            stride=sym2cpp(strides[group[-1]]),
            seed=seed_fmt.format(o=outexpr, t=output_data.dtype.ctype))
        code += stmt.format(o=outexpr, call=call) + '\n'

        code += '}\n' * len(outer_red)
        if outer_loops:
            code += '}\n' * output_dims

        tnode = dace.nodes.Tasklet('reduce', {_IN: dace.pointer(input_data.dtype)},
                                   {_OUT: dace.pointer(output_data.dtype)},
                                   code,
                                   language=dace.Language.CPP)

        inedge._dst_conn = _IN
        outedge._src_conn = _OUT
        node.add_in_connector(_IN)
        node.add_out_connector(_OUT)

        return tnode


@dace.library.expansion
class ExpandReduceCUDADevice(pm.ExpandTransformation):
    """Device-wide CUB reduce (whole array).

    Temp storage from per-stream CUB scratch pool tagged ``ReduceTag`` (see
    ``cub_scratch.cuh``, :class:`~dace.libraries.sort.environments.cub.ReduceScratch`):
    default-stream entry pre-allocated 128 MB at SDFG init, other streams lazily; reused
    per stream, grown in place on demand, freed at SDFG exit. ``__dace_current_stream``
    threads both scratch lookup and CUB call → no cross-stream race on the pool.
    """

    @classmethod
    def _resolve_environments(cls):
        # Lazy import: ReduceScratch lives in dace.libraries.sort.environments.cub, whose
        # __init__ pulls in standard.environments.cuda → avoid standard↔sort import race.
        if cls.environments is None or len(cls.environments) < 2:
            from dace.libraries.sort.environments.cub import ReduceScratch
            cls.environments = [CUDA, ReduceScratch]
        return cls.environments

    environments = [CUDA]

    _SPECIAL_RTYPES = {
        dtypes.ReductionType.Min_Location: 'ArgMin',
        dtypes.ReductionType.Max_Location: 'ArgMax',
    }

    @staticmethod
    def expansion(node: 'Reduce', state: SDFGState, sdfg: SDFG):
        from dace.codegen.prettycode import CodeIOStream
        from dace.codegen.targets.cpp import unparse_cr_split, mangle_dace_state_struct_name

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

        # Code sinks. No per-libnode init/exit: the per-stream ReduceTag pool
        # (ReduceScratch) owns scratch lifecycle for all CUB reduce calls.
        cuda_globalcode = CodeIOStream()
        host_globalcode = CodeIOStream()
        host_localcode = CodeIOStream()
        output_memlet = output_edge.data

        # Try to autodetect reduction type
        redtype = detect_reduction_type(node.wcr)

        node_id = state.node_id(node)
        state_id = state.parent_graph.node_id(state)
        idstr = '{sdfg}_{state}_{node}'.format(sdfg=sdfg.name, state=state_id, node=node_id)

        # A connector carries a type only when someone set one; ``Reduce`` declares ``_out``
        # untyped, so the array is the answer in the ordinary case. This branch was UNREACHABLE
        # while Reduce declared no connectors at all, which is how ``next(dict.values())`` -- a
        # TypeError, ``values()`` is not an iterator -- sat here undetected.
        out_conn_type = next(iter(node.out_connectors.values()), None)
        dtype = out_conn_type if out_conn_type is not None else sdfg.arrays[output_memlet.data].dtype

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
        }};""".format(id=idstr, arg1=arg1, arg2=arg2, contents=body), state.parent_graph, state_id, node_id)
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

        # Verify that the INPUT is on the GPU (CUB DeviceReduce reads a device/pinned pointer). A
        # host input cannot feed a device reduce, so fall back to the pure expansion.
        if input_data.storage not in GPU_REDUCE_DEVICE_WRITABLE_STORAGE:
            warnings.warn('Input of GPU reduction must either reside '
                          ' in global GPU memory or pinned CPU memory')
            return ExpandReducePure.expansion(node, state, sdfg)

        # Storage-aware OUTPUT: CUB writes a raw device pointer. A device-writable output
        # (GPU_Global / CPU_Pinned) is written directly (the tasklet is returned as-is). A host output
        # is NOT a reason to fall back to the slow pure loop -- instead CUB writes a GPU_Global len-1
        # scratch and a trailing device->host copy moves the result to the real output (built below).
        output_on_device = output_data.storage in GPU_REDUCE_DEVICE_WRITABLE_STORAGE

        # Determine reduction type
        kname = (ExpandReduceCUDADevice._SPECIAL_RTYPES[redtype]
                 if redtype in ExpandReduceCUDADevice._SPECIAL_RTYPES else 'Reduce')

        # Pull in per-stream CUB scratch pool env (ReduceScratch, cub_scratch.cuh).
        ExpandReduceCUDADevice._resolve_environments()

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

        # Reduce fn: query temp-storage size, fetch from per-stream ReduceTag pool
        # (lazy alloc new streams, grow in place on demand), then run CUB. Every step reports its
        # status rather than dropping it: CUB reads a null workspace as "only report the size", so a
        # failed query leaves the size unset and a failed allocation hands back a null pointer -- and
        # either turns the reduction below into a silent no-op that leaves the output untouched.
        # ``__state`` is threaded in (the caller already has it) so the query can go through the
        # standard ``DACE_GPU_CHECK`` -- which only records, never returns -- so the early ``return``
        # right after it is what actually stops a failed query from reaching CUB with a null workspace.
        state_t = mangle_dace_state_struct_name(sdfg)
        cuda_globalcode.write("""
DACE_EXPORTED gpuError_t __dace_reduce_{id}({intype} *input, {outtype} *output, {reduce_range_def}, gpuStream_t stream, {state_t} *__state);
gpuError_t __dace_reduce_{id}({intype} *input, {outtype} *output, {reduce_range_def}, gpuStream_t stream, {state_t} *__state)
{{
    size_t _cub_needed = 0;
    gpuError_t _cub_status;
    DACE_GPU_CHECK(_cub_status = gpucub::{reduce_type}::{kname}(nullptr, _cub_needed,
                                input, output, {reduce_range_use}{redop}, stream));
    if (_cub_status != gpuSuccess) return _cub_status;
    void* _cub_scratch = ::dace::cub::get_scratch<::dace::cub::ReduceTag>(_cub_needed, stream, &_cub_status);
    if (_cub_scratch == nullptr) return _cub_status != gpuSuccess ? _cub_status : gpuErrorMemoryAllocation;
    return gpucub::{reduce_type}::{kname}(_cub_scratch, _cub_needed,
                                input, output, {reduce_range_use}{redop}, stream);
}}
        """.format(id=idstr,
                   intype=input_data.dtype.ctype,
                   outtype=output_data.dtype.ctype,
                   reduce_type=reduce_type,
                   reduce_range_def=reduce_range_def,
                   reduce_range_use=reduce_range_use,
                   kname=kname,
                   redop=reduce_op,
                   state_t=state_t))

        # Write reduction function definition in caller file
        host_globalcode.write(
            """
DACE_EXPORTED gpuError_t __dace_reduce_{id}({intype} *input, {outtype} *output, {reduce_range_def}, gpuStream_t stream, {state_t} *__state);
        """.format(id=idstr,
                   reduce_range_def=reduce_range_def,
                   intype=input_data.dtype.ctype,
                   outtype=output_data.dtype.ctype,
                   state_t=state_t), state.parent_graph, state_id, node)

        # Storage-aware tasklet connector names. A device-writable output returns the tasklet
        # directly, so its connectors must be the outer ``_in`` / ``_out`` the node exposes. A host
        # output wraps the tasklet in a nested SDFG whose boundary arrays are named ``_in`` / ``_out``;
        # the tasklet's connectors must therefore NOT collide with those array names, so use
        # ``_cub_in`` / ``_cub_out`` and wire the output to the GPU scratch.
        if output_on_device:
            tin, tout = '_in', '_out'
        else:
            tin, tout = '_cub_in', '_cub_out'

        # Call reduction function where necessary
        host_localcode.write(
            'DACE_GPU_CHECK(__dace_reduce_{id}({tin}, {tout}, {reduce_range_call}, __dace_current_stream, __state));'.
            format(id=idstr, tin=tin, tout=tout, reduce_range_call=reduce_range_call))

        # Make tasklet
        tnode = dace.nodes.Tasklet('reduce', {tin: dace.pointer(input_data.dtype)},
                                   {tout: dace.pointer(output_data.dtype)},
                                   host_localcode.getvalue(),
                                   language=dace.Language.CPP)

        # Add the rest of the code (scratch init/exit lives in the env).
        sdfg.append_global_code(host_globalcode.getvalue())
        sdfg.append_global_code(cuda_globalcode.getvalue(), 'cuda')

        # Rename outer connectors and add to node

        # Device-writable output: CUB writes the output pointer directly.
        if output_on_device:
            return tnode

        # Host output: wrap the CUB tasklet in a nested SDFG that reduces into a GPU_Global scratch
        # and copies it to the real (host) ``_out`` -- a device->host copy at the storage boundary.
        # The tasklet's ``_cub_out`` connector is wired to the scratch, never to host memory.
        outsubset = dcpy(output_edge.data.subset)
        osqdim = outsubset.squeeze()
        if len(osqdim) == 0:  # scalar output
            osqdim = [0]

        nsdfg = SDFG('reduce_cub')
        nsdfg.add_array('_in',
                        insubset.size(),
                        input_data.dtype,
                        strides=[input_data.strides[orig] * insubset[j][2] for j, orig in enumerate(isqdim)],
                        storage=input_data.storage)
        nsdfg.add_transient('_out_gpu', outsubset.size(), output_data.dtype, storage=dtypes.StorageType.GPU_Global)
        nsdfg.add_array('_out',
                        outsubset.size(),
                        output_data.dtype,
                        strides=[s for i, s in enumerate(output_data.strides) if i in osqdim],
                        storage=output_data.storage)

        nstate = nsdfg.add_state()
        rin = nstate.add_read('_in')
        nstate.add_node(tnode)
        nstate.add_edge(rin, None, tnode, tin, dace.Memlet.from_array('_in', nsdfg.arrays['_in']))
        scratch_write = nstate.add_write('_out_gpu')
        nstate.add_edge(tnode, tout, scratch_write, None, dace.Memlet.from_array('_out_gpu', nsdfg.arrays['_out_gpu']))

        copy_state = nsdfg.add_state_after(nstate)
        copy_state.add_nedge(copy_state.add_read('_out_gpu'), copy_state.add_write('_out'),
                             dace.Memlet.from_array('_out', nsdfg.arrays['_out']))
        return nsdfg


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
        state_id = state.parent_graph.node_id(state)
        idstr = '{sdfg}_{state}_{node}'.format(sdfg=sdfg.name, state=state_id, node=node_id)

        # Obtain some SDFG-related information
        input_memlet = input_edge.data
        output_memlet = output_edge.data

        # A connector carries a type only when someone set one; ``Reduce`` declares ``_out``
        # untyped, so the array is the answer in the ordinary case. This branch was UNREACHABLE
        # while Reduce declared no connectors at all, which is how ``next(dict.values())`` -- a
        # TypeError, ``values()`` is not an iterator -- sat here undetected.
        out_conn_type = next(iter(node.out_connectors.values()), None)
        dtype = out_conn_type if out_conn_type is not None else sdfg.arrays[output_memlet.data].dtype
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
        }};""".format(id=idstr, arg1=arg1, arg2=arg2, contents=body), state.parent_graph, state_id, node_id)
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
        typedef gpucub::BlockReduce<{type}, {numthreads}> BlockReduce_{id};
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

        return tnode


@dace.library.expansion
class ExpandReduceCUDABlockStrided(pm.ExpandTransformation):
    """Reduce ``M`` elements with the ``B`` threads of ONE block: block-strided loop into CUB.

    :class:`ExpandReduceCUDABlock` is the one-element-per-thread form -- register in, register out,
    ``M == B``. That is not the shape a kernel presents. Measured over the ML/scientific-computing
    tracks, an in-kernel ``Reduce`` runs along a feature axis of 10^2 to 10^4 elements
    (``cross_entropy_loss``: 46,341 classes) with a 256-wide block, and until now fell back to
    ``pure`` -- ONE thread walking all of them.

    The reduced region is a one-dimensional run with a stride, which is the general case rather than
    a special one: contiguous is simply ``stride == 1``. Both forms occur (``[i, 0:num_classes]`` at
    stride 1, ``[i, 0:out_channels, j, k]`` at stride ``(h-k+1)*(w-k+1)``).

    Refuses -- and so falls back to ``pure``, slower but never wrong -- when the reduced region is
    not a single run, when the reduction is ``Custom`` or a ``*_Location`` argmin/argmax, or when the
    op has no identity to pad the final partial chunk with.
    """

    environments = [CUDA]

    @staticmethod
    def expansion(node: 'Reduce', state: SDFGState, sdfg: SDFG):
        from dace.libraries.standard.block_reduce import (BLOCK_COLLECTIVE_THREADS, add_block_lane_map,
                                                          block_reduce_code)

        node.validate(sdfg, state)
        in_edge = state.in_edges(node)[0]
        out_edge = state.out_edges(node)[0]
        in_desc = sdfg.arrays[in_edge.data.data]
        out_desc = sdfg.arrays[out_edge.data.data]

        redtype = detect_reduction_type(node.wcr)
        if redtype == dtypes.ReductionType.Custom:
            raise NotImplementedError('Reduce(CUDA (block strided)): a Custom WCR is not supported.')
        if redtype in ExpandReduceCUDABlock._SPECIAL_RTYPES:
            raise NotImplementedError(f'Reduce(CUDA (block strided)): {redtype} is not supported.')
        if out_edge.data.subset.num_elements() != 1:
            raise NotImplementedError('Reduce(CUDA (block strided)): only a full reduction to a scalar.')

        # The reduced region has to be ONE run for a strided walk to cover it exactly. ``squeeze``
        # drops the axes the subset already pinned to a point, and what is left must be a single
        # axis -- anything else would need a nested walk this emitter does not do.
        insubset = dcpy(in_edge.data.subset)
        kept = insubset.squeeze()
        if len(kept) != 1:
            raise NotImplementedError('Reduce(CUDA (block strided)): the reduced region is not a single '
                                      f'1-D run (kept axes: {kept}).')
        count = insubset.num_elements()
        stride = in_desc.strides[kept[0]]

        dtype = out_desc.dtype.base_type
        ctype = dtype.ctype
        identity = node.identity
        if identity is None:
            # A Reduce built from a WCR carries no identity of its own; the op's is well known and
            # is what the out-of-range lanes must fold.
            identity = dtypes.reduction_identity(dtype, redtype)
        if identity is None:
            raise NotImplementedError(f'Reduce(CUDA (block strided)): {redtype} has no identity to pad the '
                                      'final partial chunk with.')
        credtype = 'dace::ReductionType::' + str(redtype)[str(redtype).find('.') + 1:]
        redop = f'dace::_wcr_fixed<{credtype}, {ctype}>()'

        state_id = state.parent_graph.node_id(state)
        idstr = f'{sdfg.name}_{state_id}_{state.node_id(node)}'
        code = block_reduce_code(idstr=idstr,
                                 ctype=ctype,
                                 lanes=BLOCK_COLLECTIVE_THREADS,
                                 count_expr=symstr(count),
                                 element_expr=f'__brin[__bri * ({symstr(stride)})]',
                                 redop=redop,
                                 identity=f'static_cast<{ctype}>({symstr(identity)})',
                                 out_expr='__brout[0]')

        nsdfg = dace.SDFG(node.label + '_block')
        nsdfg.add_array('_in', [count], in_desc.dtype, strides=[stride], storage=in_desc.storage)
        nsdfg.add_array('_out', [1], out_desc.dtype, storage=out_desc.storage)
        nstate = nsdfg.add_state(node.label + '_block_state')
        tasklet = nstate.add_tasklet(node.label + '_block_reduce',
                                     inputs={'__brin': dace.pointer(in_desc.dtype.base_type)},
                                     outputs={'__brout': dace.pointer(out_desc.dtype.base_type)},
                                     code=code,
                                     language=dace.Language.CPP)
        entry, exit_node = add_block_lane_map(nstate, node.label + '_block_lanes')
        # EVERY lane sees the WHOLE run: the map supplies threads, it does not partition the data.
        nstate.add_memlet_path(nstate.add_read('_in'),
                               entry,
                               tasklet,
                               dst_conn='__brin',
                               memlet=dace.Memlet.simple('_in', f'0:{symstr(count)}'))
        nstate.add_memlet_path(tasklet,
                               exit_node,
                               nstate.add_write('_out'),
                               src_conn='__brout',
                               memlet=dace.Memlet.simple('_out', '0'))
        return nsdfg


@dace.library.expansion
class ExpandReduceCUDABlockAtomic(pm.ExpandTransformation):
    """Thread-block reduce committing ONE atomic per block to a global output.

    ``gpucub::BlockReduce`` folds the block to a single value (thread 0); thread 0 does one
    ``reduce_atomic`` into the length-1 global output. Grid-of-blocks shape for the GPU
    tile path: many blocks → one output, one atomic each (vs device-wide CUB scratch).
    Unlike :class:`ExpandReduceCUDABlock` (register output, one block per output), output
    is GPU-global length-1 so every block's atomic hits the same element. ``__shared__``
    temp storage + atomic emitted inside the tasklet → stateless ``tile -> global`` node.
    """
    environments = [CUDA]

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

        cuda_globalcode = CodeIOStream()
        localcode = CodeIOStream()

        redtype = detect_reduction_type(node.wcr)
        node_id = state.node_id(node)
        state_id = state.parent_graph.node_id(state)
        idstr = '{sdfg}_{state}_{node}'.format(sdfg=sdfg.name, state=state_id, node=node_id)

        output_memlet = output_edge.data
        # A connector carries a type only when someone set one; ``Reduce`` declares ``_out``
        # untyped, so the array is the answer in the ordinary case. This branch was UNREACHABLE
        # while Reduce declared no connectors at all, which is how ``next(dict.values())`` -- a
        # TypeError, ``values()`` is not an iterator -- sat here undetected.
        out_conn_type = next(iter(node.out_connectors.values()), None)
        dtype = out_conn_type if out_conn_type is not None else sdfg.arrays[output_memlet.data].dtype
        output_type = dtype.ctype

        if node.identity is None:
            raise ValueError('For block-atomic reduce nodes, the initial value (identity) must be specified')

        # Build the reduce functor (block fold) and pick the matching atomic.
        if redtype == dtypes.ReductionType.Custom:
            body, [arg1, arg2] = unparse_cr_split(sdfg, node.wcr)
            cuda_globalcode.write(
                """
        struct __reduce_{id} {{
            template <typename T>
            DACE_HDFI T operator()(const T &{arg1}, const T &{arg2}) const {{
                {contents}
            }}
        }};""".format(id=idstr, arg1=arg1, arg2=arg2, contents=body), state.parent_graph, state_id, node_id)
            redop = '__reduce_%s()' % idstr
            atomic = 'dace::wcr_custom<%s>::template reduce_atomic(__reduce_%s(), %%s, %%s)' % (output_type, idstr)
        else:
            credtype = 'dace::ReductionType::' + str(redtype)[str(redtype).find('.') + 1:]
            redop = 'dace::_wcr_fixed<%s, %s>()' % (credtype, output_type)
            atomic = 'dace::_wcr_fixed<%s, %s>::reduce_atomic(%%s, %%s)' % (credtype, output_type)

        # BlockReduce thread count = flattened GPU-device-map block dims (via
        # devicelevel_block_size), emitted as symbolic template arg → specializes to a
        # constant at codegen but stays symbolic-tolerant (unlike constant-only block expansion).
        block_dims = devicelevel_block_size(sdfg, state, node)
        if block_dims is None:
            raise ValueError('Block-atomic GPU reduction must occur within a GPU kernel')
        num_threads = symstr(functools.reduce(lambda a, b: a * b, block_dims, 1))
        if node.axes is not None and len(node.axes) < input_dims:
            raise ValueError('Only full reduction is supported for block-atomic reduce; use the pure expansion')
        if input_data.storage != dtypes.StorageType.Register:
            raise ValueError('Block-atomic reduction requires a GPU register input (the per-thread partial)')

        # ``_out`` addresses the global length-1 output element; take its address for the atomic.
        out_ref = cpp_array_expr(sdfg, output_memlet)
        localcode.write("""
        typedef gpucub::BlockReduce<{type}, {numthreads}> BlockReduce_{id};
        __shared__ typename BlockReduce_{id}::TempStorage temp_storage_{id};
        {type} __block_result_{id} = BlockReduce_{id}(temp_storage_{id}).Reduce({input}, {redop});
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {{
            {atomic};
        }}
            """.format(
            id=idstr,
            type=output_type,
            numthreads=num_threads,
            # One value per thread: the Register partial is length 1 by the guard above, and
            # no BlockReduce::Reduce overload takes it as the ``const T[1]`` the frame emits.
            input='*(%s)' % input_edge.data.data,
            redop=redop,
            atomic=atomic % ('&(%s)' % out_ref, '__block_result_%s' % idstr)))

        tnode = dace.nodes.Tasklet('reduce', {'_in': dace.pointer(input_data.dtype)},
                                   {'_out': dace.pointer(output_data.dtype)},
                                   localcode.getvalue(),
                                   language=dace.Language.CPP)
        sdfg.append_global_code(cuda_globalcode.getvalue(), 'cuda')
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
        graph.add_edge(u=new_entry, u_connector=None, v=reduce_node, v_connector=INPUT_CONNECTOR_NAME, memlet=memlet_in)
        graph.add_edge(u=reduce_node,
                       u_connector=OUTPUT_CONNECTOR_NAME,
                       v=new_exit,
                       v_connector=None,
                       memlet=memlet_out)

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
        state_id = state.block_id
        local_storage.setup_match(sdfg, state.parent_graph.cfg_id, state_id, in_local_storage_subgraph, 0)

        local_storage.array = in_edge.data.data
        local_storage.apply(graph, sdfg)
        in_transient = local_storage._data_node
        sdfg.data(in_transient.data).storage = dtypes.StorageType.Register

        local_storage = OutLocalStorage()
        local_storage.setup_match(sdfg, state.parent_graph.cfg_id, state_id, out_local_storage_subgraph, 0)
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
        sub_expansion.setup_match(sdfg, state.parent_graph.cfg_id, state_id, {}, 0)
        return sub_expansion.expansion(node=node, state=state, sdfg=sdfg)
        #return reduce_node.expand(state)


@dace.library.expansion
class ExpandReduceGPUAuto(pm.ExpandTransformation):
    """
        GPU implementation of the reduce node. This expansion aims to map the reduction inputs to an optimal GPU schedule.
    """
    environments = []

    @staticmethod
    def expansion(node: 'Reduce', state: SDFGState, sdfg: SDFG):
        """
        Expands the Reduce node.

        :param node: the node to expand
        :param state: the state in which the node is in
        :param sdfg: the SDFG in which the node is in
        """
        from dace.codegen import common

        # This expansion emits inline device code needing no external environment. Reset here so a
        # previous Custom-WCR delegation (which sets the CUB scratch environments below) does not leak
        # its 128 MB scratch-pool environment onto this non-Custom expansion. ``ExpandTransformation``
        # attaches ``type(self).environments`` to the expanded node, so the value at the end of this
        # method call is the one that takes effect.
        ExpandReduceGPUAuto.environments = []

        node.validate(sdfg, state)
        inedge: graph.MultiConnectorEdge = state.in_edges(node)[0]
        outedge: graph.MultiConnectorEdge = state.out_edges(node)[0]
        insubset = dcpy(inedge.data.subset)
        isqdim = insubset.squeeze()
        raw_input_data = sdfg.arrays[inedge.data.data]
        raw_output_data = sdfg.arrays[outedge.data.data]
        warp_size = 64 if common.get_gpu_backend() == 'hip' else 32

        in_type = raw_input_data.dtype

        if raw_input_data.storage != dtypes.StorageType.GPU_Global:
            # data doesnt reside on GPU --> return pure expansion
            warnings.warn(
                'Cannot use GPUAuto expansion: Input data does not reside on GPU. Falling back to Pure expansion')
            return ExpandReducePure.expansion(node, state, sdfg)

        if scope.is_devicelevel_gpu_kernel(sdfg, state, node):
            # Reduce node is already in a GPU kernel
            warnings.warn(
                'Cannot use GPUAuto expansion: Node to expand is already inside a GPU kernel. Falling back to Pure expansion'
            )
            return ExpandReducePure.expansion(node, state, sdfg)

        if node.identity is None:
            warnings.warn('Cannot use GPUAuto expansion: node.identity is None. Falling back to Pure expansion')
            return ExpandReducePure.expansion(node, state, sdfg)

        # A Custom WCR (e.g. an ITE ``a if a > b else b`` -- argmax/argmin-style tiebreak) cannot be
        # expressed through the ``dace::warpReduce<ReductionType, T>`` primitive this expansion uses
        # (that primitive is templated on a FIXED reduction type, with no Custom variant). Rather than
        # raise or fall back to the slow pure loop, delegate to the CUB ``DeviceReduce`` expansion,
        # which lowers a Custom op as a device functor (``struct __reduce_{id}``) -- an efficient
        # device reduction that is storage-aware (host output via a GPU scratch + copy). Identity is
        # guaranteed set here (checked above), which CUB device requires.
        if detect_reduction_type(node.wcr) == dtypes.ReductionType.Custom:
            # The CUB DeviceReduce functor path emits ``::dace::cub::get_scratch<ReduceTag>`` and needs
            # the per-stream scratch-pool environment (ReduceScratch). ``ExpandTransformation.apply``
            # attaches THIS transformation's ``environments`` to the delegated expansion, so carry the
            # CUB scratch environments here (reset to ``[]`` for the non-Custom path above).
            ExpandReduceGPUAuto.environments = list(ExpandReduceCUDADevice._resolve_environments())
            return ExpandReduceCUDADevice.expansion(node, state, sdfg)

        # Standardize and squeeze axes
        axes = node.axes if node.axes is not None else [i for i in range(len(inedge.data.subset))]
        # this removes reduction of size 1 axes from the list
        axes = [axis for axis in axes if axis in isqdim]

        # When the reduce reads a View (e.g. the sliced ``x[:, 2*i:2*i+2, ...]`` window feeding a
        # maxpool, or a reshaped array), a View is ``source_ptr + offset`` with its own strides/step.
        # Resolve it to a plain array describing the actual viewed region: fold the input memlet
        # subset's step into the underlying strides (exactly like :class:`ExpandReducePure`). This lets
        # the planner schedule the efficient device reduction directly over the (possibly strided) view,
        # and keeps ``_in`` a plain array rather than a bare View access node (which is an invalid
        # nested-SDFG input connector with no defining view edge). Non-View inputs are handed to the
        # planner unchanged.
        if isinstance(raw_input_data, dace.data.View):
            in_subset = inedge.data.subset
            planner_input = dace.data.Array(
                raw_input_data.dtype,
                in_subset.size(),
                storage=raw_input_data.storage,
                strides=[raw_input_data.strides[i] * in_subset[i][2] for i in range(len(in_subset))])
        else:
            planner_input = raw_input_data

        # call the planner script
        schedule = red_planner.get_reduction_schedule(planner_input, axes, warp_size=warp_size)

        if schedule.error:
            # return pure expansion if error
            warnings.warn(schedule.error)
            pure_sdfg = ExpandReducePure.expansion(node, state, sdfg)
            return pure_sdfg

        # Create nested SDFG
        nsdfg = SDFG('reduce')

        input_data = dcpy(planner_input)
        input_data.transient = False
        input_data.shape = schedule.in_shape
        input_data.strides = schedule.in_strides
        # The planner may flatten the rank (e.g. (M, N, K) -> (M*N, K)); the copied offset keeps the
        # OLD rank and descriptor validation rejects the mismatch at the next add_view/deepcopy.
        input_data.offset = [0] * len(schedule.in_shape)
        nsdfg.add_datadesc('_in', input_data)

        output_data = dcpy(raw_output_data)
        # The reduction body writes ``_out`` from GPU device maps, so ``_out`` must be device-writable.
        # If the real output already lives in device memory, ``_out`` IS that output (written directly).
        # If it lives on the host (Register / CPU_Heap / Default / ...), ``_out`` is built as a
        # GPU_Global scratch here and, once the reduction body is complete, renamed to a transient and
        # copied to the real host output by :func:`route_gpu_reduce_result_to_host_output` -- so the GPU
        # kernels never write host memory (and no slow pure fallback is taken).
        output_on_device = raw_output_data.storage in GPU_REDUCE_DEVICE_WRITABLE_STORAGE

        # ``_out``'s declared strides must match the memory it is written into. The planner derives
        # ``out_strides`` from the INPUT array, which is only a valid layout for the reconciled host
        # scratch. When ``_out`` IS the real device output (written directly) and that output is a
        # strided View (e.g. ``output[:, i, j, :]``), those input-derived strides send the writes to
        # the wrong offsets, so use the real output's strides for the written (squeezed) region.
        out_strides = schedule.out_strides
        if output_on_device:
            osqdim = dcpy(outedge.data.subset).squeeze()
            if len(osqdim) == len(schedule.out_shape):
                out_strides = [raw_output_data.strides[i] for i in osqdim]
        nsdfg.add_array('_out',
                        schedule.out_shape,
                        output_data.dtype,
                        strides=out_strides,
                        storage=(output_data.storage if output_on_device else dtypes.StorageType.GPU_Global))

        nstate = nsdfg.add_state()

        # Interleave input and output axes to match input memlet
        ictr, octr, actr = 0, 0, 0
        input_subset = []
        dims = list(range(len(schedule.in_shape)))
        for i in dims:
            if i in schedule.axes:
                if i == schedule.axes[-1]:
                    input_subset.append('_i%d' % ictr)
                    ictr += 1
                else:
                    input_subset.append('_a%d' % actr)
                    actr += 1
            else:
                input_subset.append('_o%d' % octr)
                octr += 1

        vectorize = schedule.vectorize
        mini_warps = schedule.mini_warps

        # produce the SDFG depending on schedule.contiguous_dim
        if schedule.contiguous_dim:
            # we are reducing the contiguous dimension

            outm = dace.Memlet(f'_out[{",".join(["_o%d" % i for i in range(len(schedule.out_shape))])}]', dynamic=True)
            outm_wcr = dace.Memlet(f'_out[{",".join(["_o%d" % i for i in range(len(schedule.out_shape))])}]',
                                   dynamic=True,
                                   wcr=node.wcr)
            inmm = dace.Memlet(f'_in[{",".join(input_subset)}]')

            if schedule.one_d_reduction:
                outm = dace.Memlet(f'_out[0]', dynamic=True, wcr=node.wcr)

                # initialize output to zero
                init_state = nsdfg.add_state()
                nsdfg.add_edge(init_state, nstate, dace.InterstateEdge())

                # Add initialization as a map
                init_state.add_mapped_tasklet('reduce_init', {'_o': '0:1'}, {},
                                              '__out = %s' % node.identity, {'__out': dace.Memlet('_out[0]')},
                                              external_edges=True,
                                              schedule=dtypes.ScheduleType.GPU_Device)

            if schedule.multi_axes:
                # initialize output to zero
                init_state = nsdfg.add_state()
                nsdfg.add_edge(init_state, nstate, dace.InterstateEdge())

                # Add initialization as a map
                init_state.add_mapped_tasklet('reduce_init', {
                    f'_o{i}': subsets.Range([(0, sz - 1, 1)])
                    for i, sz in enumerate(schedule.out_shape)
                }, {},
                                              '__out = %s' % node.identity, {'__out': outm},
                                              external_edges=True,
                                              schedule=dtypes.ScheduleType.GPU_Device)

                # additional grid dims
                add_me, add_mx = nstate.add_map('grid', {
                    f'_a{i}': subsets.Range([(0, sz - 1, 1)])
                    for i, sz in enumerate(schedule.additional_grid)
                },
                                                schedule=dtypes.ScheduleType.GPU_Device)

            # add map, which corresponds to the CUDA grid
            ome, omx = nstate.add_map('grid', {
                f'_o{i}': subsets.Range([(0, sz - 1, 1)])
                for i, sz in enumerate(schedule.grid)
            },
                                      schedule=dtypes.ScheduleType.GPU_Device)

            # add map, which corresponds to the thread blocks
            bme, bmx = nstate.add_map('thread_block', {'tid': subsets.Range([(0, sz - 1, 1)])
                                                       for sz in schedule.block},
                                      schedule=dtypes.ScheduleType.GPU_ThreadBlock)

            if vectorize:
                nsdfg.add_scalar('acc_vec', dace.vector(in_type, schedule.vec_len), dtypes.StorageType.Register, True)
                acc_vec_1 = nstate.add_access('acc_vec')
                acc_vec_2 = nstate.add_access('acc_vec')

                if schedule.vec_len == 2:
                    init_vec = nstate.add_tasklet('init_vec', {}, {'__o_out'},
                                                  f'__o_out.x = {node.identity}\n__o_out.y = {node.identity}')
                elif schedule.vec_len == 4:
                    init_vec = nstate.add_tasklet(
                        'init_vec', {}, {'__o_out'},
                        f'__o_out.x = {node.identity}\n__o_out.y = {node.identity}\n__o_out.z = {node.identity}\n__o_out.w = {node.identity}'
                    )
                else:
                    raise ValueError(f'Vector length of {schedule.vec_len} not supported')

                nstate.add_edge(bme, None, init_vec, None, dace.Memlet())
                nstate.add_edge(init_vec, '__o_out', acc_vec_1, None, dace.Memlet('acc_vec'))

            nsdfg.add_scalar('acc', nsdfg.arrays['_in'].dtype, dtypes.StorageType.Register, True)
            acc_1 = nstate.add_access('acc')
            acc_2 = nstate.add_access('acc')
            acc_3 = nstate.add_access('acc')

            init_scalar = nstate.add_tasklet('init_scalar', {}, {'__o_out'}, f'__o_out = {node.identity}')
            nstate.add_edge(bme, None, init_scalar, None, dace.Memlet())
            nstate.add_edge(init_scalar, '__o_out', acc_1, None, dace.Memlet('acc[0]'))

            # Add inner map, which corresponds to the range to reduce, containing an identity tasklet
            # with vectorization we simply have different start and stride
            if schedule.one_d_reduction:
                ime, imx = nstate.add_map('reduce_values', {
                    '_j0':
                    subsets.Range([(f'_o0*1024', schedule.in_shape[0] - 1, 1024 * schedule.grid[0])]),
                    '_i0':
                    subsets.Range([(f'{schedule.vec_len if vectorize else 1}*tid+_j0',
                                    f'Min(_j0+1023, {schedule.in_shape[0]-1})', schedule.sequential[0][2])])
                },
                                          schedule=dtypes.ScheduleType.Sequential)
            else:
                ime, imx = nstate.add_map(
                    'reduce_values', {
                        f'_i{i}': subsets.Range([(f'{schedule.vec_len if vectorize else 1}*tid', s[1] - 1, s[2])])
                        for i, s in enumerate(schedule.sequential)
                    },
                    schedule=dtypes.ScheduleType.Sequential)

            # Add identity tasklet for reduction
            if vectorize:
                id = nstate.add_tasklet('identity', {
                    '__a_in': dace.vector(in_type, schedule.vec_len),
                    '__b_in': dace.vector(in_type, schedule.vec_len)
                }, {'__o_out'}, '__o_out = __b_in')
            else:
                id = nstate.add_tasklet('identity', {'__a_in', '__b_in'}, {'__o_out'}, '__o_out = __b_in')

            if vectorize:
                # add a vec_reduce tasklet
                vr = nstate.add_tasklet('vec_reduce', {
                    '__a_in': in_type,
                    '__b_in': dace.vector(in_type, schedule.vec_len)
                }, {'__o_out': dace.vector(in_type, schedule.vec_len)}, '__o_out = __b_in')

            # add warpReduce tasklet
            ctype = output_data.dtype
            redtype = detect_reduction_type(node.wcr)
            if redtype == dtypes.ReductionType.Custom:
                # Unreachable: a Custom WCR is delegated to the CUB DeviceReduce functor path at the
                # top of this expansion (``dace::warpReduce`` has no Custom variant).
                raise NotImplementedError('Custom WCR must be delegated to ExpandReduceCUDADevice; '
                                          'reached the warpReduce path unexpectedly')
            credtype = ('dace::ReductionType::' + str(redtype)[str(redtype).find('.') + 1:])
            wr = nstate.add_tasklet('warp_reduce', {'__a'}, {'__out'},
                                    f'__out = dace::warpReduce<{credtype}, {ctype}>::reduce(__a);', dtypes.Language.CPP)

            cond_tasklet = nstate.add_tasklet('cond_write', {'_input'}, {'_output'},
                                              'if threadIdx.x == 0: _output = _input')

            # Connect everything
            r = nstate.add_read('_in')
            w = nstate.add_write('_out')

            if schedule.multi_axes:
                nstate.add_memlet_path(r, add_me, ome, bme, ime, id, dst_conn='__b_in', memlet=inmm)
            else:
                nstate.add_memlet_path(r, ome, bme, ime, id, dst_conn='__b_in', memlet=inmm)

            if vectorize:
                nstate.add_memlet_path(acc_vec_1, ime, id, dst_conn='__a_in', memlet=dace.Memlet('acc_vec[0]'))
                nstate.add_memlet_path(id,
                                       imx,
                                       acc_vec_2,
                                       src_conn='__o_out',
                                       memlet=dace.Memlet('acc_vec[0]', wcr=node.wcr))
                nstate.add_memlet_path(acc_vec_2, vr, dst_conn='__b_in', memlet=dace.Memlet('acc_vec[0]'))
                nstate.add_memlet_path(acc_1, vr, dst_conn='__a_in', memlet=dace.Memlet('acc[0]'))
                nstate.add_memlet_path(vr, acc_2, src_conn='__o_out', memlet=dace.Memlet('acc[0]', wcr=node.wcr))
            else:
                nstate.add_memlet_path(acc_1, ime, id, dst_conn='__a_in', memlet=dace.Memlet('acc[0]'))
                nstate.add_memlet_path(id, imx, acc_2, src_conn='__o_out', memlet=dace.Memlet('acc[0]', wcr=node.wcr))

            nstate.add_memlet_path(acc_2, wr, dst_conn='__a', memlet=dace.Memlet('acc[0]'))
            nstate.add_memlet_path(wr, bmx, acc_3, src_conn='__out', memlet=dace.Memlet('acc[0]'))

            nstate.add_memlet_path(acc_3, cond_tasklet, dst_conn='_input', memlet=dace.Memlet('acc[0]'))
            if schedule.multi_axes:
                nstate.add_memlet_path(cond_tasklet, omx, add_mx, w, src_conn='_output', memlet=outm_wcr)
            else:
                nstate.add_memlet_path(cond_tasklet, omx, w, src_conn='_output', memlet=outm)

        else:  # we are reducing a non-contiguous dimension

            nested_sdfg = dace.SDFG('nested_sdfg')
            start_state = nested_sdfg.add_state('start_state')
            real_state = nested_sdfg.add_state('real_state')

            nested_sdfg.add_edge(start_state, real_state,
                                 dace.InterstateEdge(f'_b1 + {warp_size} * _g < {schedule.in_shape[-1]}'))

            reset_outm = dace.Memlet(f'_out[{",".join(["_o%d" % i for i in range(len(schedule.out_shape))])}]')
            if len(schedule.out_shape) > 1:
                outm = dace.Memlet(
                    f'_out[{",".join(["_o%d" % i for i in range(len(schedule.out_shape) - 1)])},_g * {warp_size} + _b]',
                    dynamic=True)
                outm_wcr = dace.Memlet(
                    f'_out[{",".join(["_o%d" % i for i in range(len(schedule.out_shape) - 1)])},_g * {warp_size} + _b]',
                    dynamic=True,
                    wcr=node.wcr)

            else:
                outm = dace.Memlet(f'_out[_g * {warp_size} + _b]', dynamic=True)
                outm_wcr = dace.Memlet(f'_out[_g * {warp_size} + _b]', dynamic=True, wcr=node.wcr)

            input_subset = input_subset[:-2]
            input_subset.append(f'0:{schedule.sequential[0]}')
            input_subset.append(f'_g * {warp_size} + _b1')
            inmm = dace.Memlet(f'_in[{",".join(input_subset)}]', dynamic=True)

            if schedule.multi_axes:
                # Add initialization
                init_state = nsdfg.add_state()
                nsdfg.add_edge(init_state, nstate, dace.InterstateEdge())
                init_state.add_mapped_tasklet('reduce_init', {
                    f'_o{i}': subsets.Range([(0, sz - 1, 1)])
                    for i, sz in enumerate(schedule.out_shape)
                }, {},
                                              '__out = %s' % node.identity, {'__out': reset_outm},
                                              external_edges=True,
                                              schedule=dtypes.ScheduleType.GPU_Device)

                # additional grid dims
                add_me, add_mx = nstate.add_map('grid', {
                    f'_a{i}': subsets.Range([(0, sz - 1, 1)])
                    for i, sz in enumerate(schedule.additional_grid)
                },
                                                schedule=dtypes.ScheduleType.GPU_Device)

            if len(schedule.grid) == 1:
                ome, omx = nstate.add_map('grid', {'_g': f'0:{schedule.grid[0]}'},
                                          schedule=dtypes.ScheduleType.GPU_Device)

            else:
                grid_dict = {f'_o{i}': f'0:{sz}' for i, sz in enumerate(schedule.grid[:-1])}
                grid_dict.update({'_g': f'0:{schedule.grid[-1]}'})
                ome, omx = nstate.add_map('grid', grid_dict, schedule=dtypes.ScheduleType.GPU_Device)

            if mini_warps:
                bme1, bmx1 = nstate.add_map('block', {'_b': f'0:{schedule.block[1]}'},
                                            schedule=dtypes.ScheduleType.GPU_ThreadBlock)

                bme2, bmx2 = nstate.add_map('block', {
                    '_b0': f'0:{schedule.block[0]}',
                    '_mwid': f'0:{schedule.num_mini_warps}',
                    '_b1': f'0:{schedule.block[1]}'
                },
                                            schedule=dtypes.ScheduleType.GPU_ThreadBlock)

            else:
                bme1, bmx1 = nstate.add_map('block', {'_b': f'0:{warp_size}'},
                                            schedule=dtypes.ScheduleType.GPU_ThreadBlock)

                bme2, bmx2 = nstate.add_map('block', {
                    f'_b{i}': f'0:{sz}'
                    for i, sz in enumerate(schedule.block)
                },
                                            schedule=dtypes.ScheduleType.GPU_ThreadBlock)

            # add shared memory of warp size to outer sdfg
            nsdfg.add_array('s_mem', [schedule.shared_mem_size],
                            nsdfg.arrays['_in'].dtype,
                            dtypes.StorageType.GPU_Shared,
                            transient=True)
            s_mem1 = nstate.add_access('s_mem')
            nstate.add_edge(ome, None, s_mem1, None, dace.Memlet())

            nested_sdfg.add_scalar('s_mem', nsdfg.arrays['_in'].dtype, dtypes.StorageType.GPU_Shared)
            if schedule.multi_axes:
                nested_sdfg.add_array('_in', [schedule.sequential[0]],
                                      nsdfg.arrays['_in'].dtype,
                                      dtypes.StorageType.GPU_Global,
                                      strides=[schedule.changed_in_strides[schedule.changed_axes[0]]])
            else:
                nested_sdfg.add_array('_in', [schedule.sequential[0]],
                                      nsdfg.arrays['_in'].dtype,
                                      dtypes.StorageType.GPU_Global,
                                      strides=[schedule.in_strides[schedule.axes[0]]])

            # thread local accumulator in nested sdfg
            nested_sdfg.add_scalar('acc', nsdfg.arrays['_in'].dtype, dtypes.StorageType.Register, True)
            accread = real_state.add_access('acc')
            accwrite = real_state.add_access('acc')
            final_inner_smem = real_state.add_access('s_mem')

            init_scalar = real_state.add_tasklet('reset_acc', {}, {'__o_out'}, f'__o_out = {node.identity}')
            real_state.add_edge(init_scalar, '__o_out', accread, None, dace.Memlet('acc'))

            init_smem = nstate.add_tasklet('reset_smem', {'__a_in'}, {'__o_out'}, f'__o_out = {node.identity}')
            s_mem2 = nstate.add_access('s_mem')

            nstate.add_memlet_path(s_mem1, bme1, init_smem, dst_conn='__a_in', memlet=dace.Memlet('s_mem[_b]'))
            nstate.add_memlet_path(init_smem, bmx1, s_mem2, src_conn='__o_out', memlet=dace.Memlet('s_mem[_b]'))

            s_mem3 = nstate.add_access('s_mem')

            # Add inner map, which corresponds to the range to reduce, containing an identity tasklet
            if mini_warps:
                ime, imx = real_state.add_map('reduce_values', {
                    '_i':
                    f'_b0*{schedule.num_mini_warps}+_mwid:{schedule.sequential[0]}:{16*schedule.num_mini_warps}'
                },
                                              schedule=dtypes.ScheduleType.Sequential)
            else:
                ime, imx = real_state.add_map('reduce_values', {'_i': f'_b0:{schedule.sequential[0]}:16'},
                                              schedule=dtypes.ScheduleType.Sequential)

            id = real_state.add_tasklet('identity', {'__a_in', '__b_in'}, {'__o_out'}, '__o_out = __b_in')
            # tasklet for reducing partial results to shared memory
            id_smem = real_state.add_tasklet('identity_smem', {'__a_in', '__b_in'}, {'__o_out'}, '__o_out = __b_in')

            # Connect everything
            r = nstate.add_read('_in')
            w = nstate.add_write('_out')

            actual_nested_sdfg = nstate.add_nested_sdfg(nested_sdfg, {'s_mem', '_in'}, {'s_mem'})

            inner_in = real_state.add_access('_in')
            inner_smem = real_state.add_access('s_mem')

            if schedule.multi_axes:
                nstate.add_memlet_path(r, add_me, ome, bme2, actual_nested_sdfg, dst_conn='_in', memlet=inmm)
            else:
                nstate.add_memlet_path(r, ome, bme2, actual_nested_sdfg, dst_conn='_in', memlet=inmm)

            nstate.add_memlet_path(s_mem2, bme2, actual_nested_sdfg, dst_conn='s_mem', memlet=dace.Memlet('s_mem[_b1]'))

            nstate.add_memlet_path(actual_nested_sdfg, bmx2, s_mem3, src_conn='s_mem', memlet=dace.Memlet('s_mem[_b1]'))

            real_state.add_memlet_path(inner_in, ime, id, dst_conn='__b_in', memlet=dace.Memlet('_in[_i]'))

            if mini_warps:
                cond_tasklet = nstate.add_tasklet(
                    'cond_write', {'_input'}, {'_output'},
                    f'if _b + {warp_size} * _g < {schedule.out_shape[-1]} and _bb == 0 and _mwid == 0: _output = _input'
                )
            else:
                cond_tasklet = nstate.add_tasklet(
                    'cond_write', {'_input'}, {'_output'},
                    f'if _b + {warp_size} * _g < {schedule.out_shape[-1]} and _bb == 0: _output = _input')

            # connect accumulator to identity tasklet
            real_state.add_memlet_path(accread, ime, id, dst_conn='__a_in', memlet=dace.Memlet('acc[0]'))
            # connect output of id tasklet
            real_state.add_memlet_path(id,
                                       imx,
                                       accwrite,
                                       src_conn='__o_out',
                                       memlet=dace.Memlet('acc[0]', wcr=node.wcr))

            # connect to and from smem reduction tasklet
            real_state.add_memlet_path(inner_smem, id_smem, dst_conn='__a_in', memlet=dace.Memlet('s_mem[0]'))
            real_state.add_memlet_path(accwrite, id_smem, dst_conn='__b_in', memlet=dace.Memlet('acc[0]'))
            real_state.add_memlet_path(id_smem,
                                       final_inner_smem,
                                       src_conn='__o_out',
                                       memlet=dace.Memlet('s_mem[0]', wcr=node.wcr))

            if mini_warps:
                bme3, bmx3 = nstate.add_map('block', {
                    '_bb': f'0:{schedule.block[0]}',
                    '_mwid': f'0:{schedule.num_mini_warps}',
                    '_b': f'0:{schedule.block[1]}'
                },
                                            schedule=dtypes.ScheduleType.GPU_ThreadBlock)
                nstate.add_memlet_path(s_mem3, bme3, cond_tasklet, dst_conn='_input', memlet=dace.Memlet('s_mem[_b]'))
            else:
                bme3, bmx3 = nstate.add_map('block', {
                    '_bb': f'0:{512//warp_size}',
                    '_b': f'0:{warp_size}'
                },
                                            schedule=dtypes.ScheduleType.GPU_ThreadBlock)
                nstate.add_memlet_path(s_mem3, bme3, cond_tasklet, dst_conn='_input', memlet=dace.Memlet('s_mem[_b]'))

            if schedule.multi_axes:
                nstate.add_memlet_path(cond_tasklet, bmx3, omx, add_mx, w, src_conn='_output', memlet=outm_wcr)
            else:
                nstate.add_memlet_path(cond_tasklet, bmx3, omx, w, src_conn='_output', memlet=outm)

        # Host output: the reduction wrote a GPU_Global ``_out`` scratch; route it to the real
        # (host) output via a device->host copy so a device kernel never writes host memory.
        if not output_on_device:
            route_gpu_reduce_result_to_host_output(nsdfg, nstate, output_data.dtype, schedule.out_shape,
                                                   schedule.out_strides, raw_output_data.storage)

        # Rename outer connectors and add to node

        from dace.transformation import dataflow
        # ``validate=False``: this SDFG is still DETACHED, so nothing here can see the scope it is
        # about to be placed in. A device-resident operand read by the Sequential maps above then
        # reads as a host access of GPU memory -- which it is not, because the node these maps
        # replace is inside a kernel. The graph is validated in context by the caller.
        nsdfg.apply_transformations_repeated(dataflow.MapCollapse, validate=False)

        return nsdfg


@dace.library.node
class Reduce(dace.sdfg.nodes.LibraryNode):
    """ An SDFG node that reduces an N-dimensional array to an
        (N-k)-dimensional array, with a list of axes to reduce and
        a reduction binary function. """

    #: The node's data connectors, spelled the way ``CopyLibraryNode`` and ``Scan`` spell theirs.
    #: Named here rather than at each expansion so a caller wiring a ``Reduce`` and an expansion
    #: consuming one agree by construction.
    INPUT_CONNECTOR_NAME = INPUT_CONNECTOR_NAME
    OUTPUT_CONNECTOR_NAME = OUTPUT_CONNECTOR_NAME

    # Global properties
    implementations = {
        'auto': ExpandReduceAuto,
        'pure': ExpandReducePure,
        'pure-seq': ExpandReducePureSequentialDim,
        'OpenMP': ExpandReduceOpenMP,
        'CUDA (device)': ExpandReduceCUDADevice,
        'CUDA (block)': ExpandReduceCUDABlock,
        'CUDA (block strided)': ExpandReduceCUDABlockStrided,
        'CUDA (block atomic)': ExpandReduceCUDABlockAtomic,
        'CUDA (block allreduce)': ExpandReduceCUDABlockAll,
        'GPUAuto': ExpandReduceGPUAuto
        # 'CUDA (warp)': ExpandReduceCUDAWarp,
        # 'CUDA (warp allreduce)': ExpandReduceCUDAWarpAll
    }

    default_implementation = 'auto'

    # Properties
    axes = ListProperty(element_type=int, allow_none=True)
    wcr = LambdaProperty(default='lambda a, b: a')
    identity = Property(allow_none=True, to_json=lambda x: str(x))

    def __init__(self,
                 name,
                 wcr='lambda a, b: a',
                 axes=None,
                 identity=None,
                 schedule=dtypes.ScheduleType.Default,
                 debuginfo=None,
                 **kwargs):
        # Declared like every other library node's: data flows in through ``_in`` and out through
        # ``_out``, and an expansion replaces the node with a tasklet or nested SDFG carrying the
        # same two. Reduce used to declare neither, so each of its expansions renamed the edges at
        # expansion time -- the same three lines copied into eight places.
        super().__init__(name=name, inputs={INPUT_CONNECTOR_NAME}, outputs={OUTPUT_CONNECTOR_NAME}, **kwargs)
        self.wcr = wcr
        self.axes = axes
        self.identity = identity
        self.debuginfo = debuginfo
        self.schedule = schedule

    @staticmethod
    def from_json(json_obj, context=None):
        ret = Reduce('reduce', 'lambda a, b: a', None)
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
