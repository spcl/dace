# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains the GPUMultiTransformMap transformation. """
import dace
from dace import dtypes, registry
from dace.data import Scalar
from dace.sdfg import utils as sdutil
from dace.sdfg import nodes, SDFG, SDFGState, infer_types
from dace.properties import make_properties, Property, SymbolicProperty
from dace.transformation import transformation
from dace.sdfg import has_dynamic_map_inputs
from dace.properties import make_properties
from dace.config import Config
from dace.symbolic import SymbolicType
from dace.frontend import operations
from typing import Dict, Any
import warnings


@registry.autoregister_params(singlestate=True)
@make_properties
class GPUMultiTransformMap(transformation.Transformation):
    """ Implements the GPUMultiTransformMap transformation.

        Tiles a single map into 2 maps. The outer map is of schedule 
        GPU_Multidevice and loops over the GPUs, while the inner map
        is a GPU-scheduled map. It also creates GPU transient arrays
        between the two maps.
    """
    _map_entry = nodes.MapEntry(nodes.Map("", [], []))
    dim_idx = Property(dtype=int,
                       default=-1,
                       desc="Index of dimension to be distributed.")

    new_dim_prefix = Property(dtype=str,
                              default="gpu",
                              allow_none=True,
                              desc="Prefix for new dimension name")

    new_transient_prefix = Property(dtype=str,
                                    default="gpu_multi",
                                    allow_none=True,
                                    desc="Prefix for the transient name")

    skip_scalar = Property(
        dtype=bool,
        default=True,
        allow_none=True,
        desc="If True: skips the scalar data nodes. "
        "If False: creates localstorage for scalar transients.")

    use_p2p = Property(
        dtype=bool,
        default=False,
        allow_none=True,
        desc="If True: uses peer-to-peer access if a data container is already "
        "located on a GPU. "
        "If False: creates transient localstorage for data located on GPU.")

    number_of_gpus = SymbolicProperty(
        default=None,
        allow_none=True,
        desc="number of gpus to divide the map onto,"
        " if not used, uses the amount specified"
        " in the dace.config in max_number_gpus.")

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(GPUMultiTransformMap._map_entry)]

    @staticmethod
    def can_be_applied(graph: SDFGState,
                       candidate,
                       expr_index,
                       sdfg,
                       strict=False):
        map_entry = graph.nodes()[candidate[GPUMultiTransformMap._map_entry]]

        # Check if there is more than one GPU available:
        if (Config.get("compiler", "cuda", "max_number_gpus") < 2):
            return False

        # Dynamic map ranges not supported
        if has_dynamic_map_inputs(graph, map_entry):
            return False

        # Only accept maps with a default schedule
        schedule_whitelist = [dtypes.ScheduleType.Default]
        sdict = graph.scope_dict()
        parent = sdict[map_entry]
        while parent is not None:
            if parent.map.schedule not in schedule_whitelist:
                return False
            parent = sdict[parent]

        # Library nodes inside the scope are not supported
        scope_subgraph = graph.scope_subgraph(map_entry)
        for node in scope_subgraph.nodes():
            if isinstance(node, nodes.LibraryNode):
                return False

        # Custom reductions can not have an accumulate transient, as the
        # reduction would have to be split up for the ingoing memlet of the
        # accumulate transient and the outgoing memlet. Not using GPU local
        # accumulate transient only works for a small volume of data.
        map_exit = graph.exit_node(map_entry)
        for edge in graph.out_edges(map_exit):
            if edge.data.wcr is not None and operations.detect_reduction_type(
                    edge.data.wcr) == dtypes.ReductionType.Custom:
                return False

        storage_whitelist = [
            dtypes.StorageType.Default,
            dtypes.StorageType.CPU_Pinned,
            dtypes.StorageType.CPU_Heap,
            dtypes.StorageType.GPU_Global,
        ]
        for node in graph.predecessors(map_entry):
            if not isinstance(node, nodes.AccessNode):
                return False
            if node.desc(graph).storage not in storage_whitelist:
                return False

        for node in graph.successors(map_exit):
            if not isinstance(node, nodes.AccessNode):
                return False
            if node.desc(graph).storage not in storage_whitelist:
                return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[GPUMultiTransformMap._map_entry]]

        return map_entry.map.label

    def apply(self, sdfg: SDFG) -> None:
        graph: SDFGState = sdfg.nodes()[self.state_id]

        inner_map_entry: nodes.MapEntry = graph.nodes()[self.subgraph[
            GPUMultiTransformMap._map_entry]]

        number_of_gpus = self.number_of_gpus
        ngpus = Config.get("compiler", "cuda", "max_number_gpus")
        if (number_of_gpus == None):
            number_of_gpus = ngpus
        if number_of_gpus > ngpus:
            raise ValueError(
                'Requesting more gpus than specified in the dace config')

        # Avoiding import loops
        from dace.transformation.dataflow import (StripMining, InLocalStorage,
                                                  OutLocalStorage,
                                                  AccumulateTransient)

        # The user has responsibility for the implementation of a Library node.
        scope_subgraph = graph.scope_subgraph(inner_map_entry)
        for node in scope_subgraph.nodes():
            if isinstance(node, nodes.LibraryNode):
                warnings.warn(
                    'Node %s is a library node, make sure to manually set the '
                    'implementation to a GPU compliant specialization.' % node)

        # Tile map into number_of_gpus tiles
        outer_map: nodes.Map = StripMining.apply_to(
            sdfg,
            dict(dim_idx=-1,
                 new_dim_prefix=self.new_dim_prefix,
                 tile_size=number_of_gpus,
                 tiling_type=dtypes.TilingType.NumberOfTiles),
            _map_entry=inner_map_entry)

        outer_map_entry: nodes.MapEntry = graph.scope_dict()[inner_map_entry]
        inner_map_exit: nodes.MapExit = graph.exit_node(inner_map_entry)
        outer_map_exit: nodes.MapExit = graph.exit_node(outer_map_entry)

        # Change map schedules
        inner_map_entry.map.schedule = dtypes.ScheduleType.GPU_Device
        outer_map.schedule = dtypes.ScheduleType.GPU_Multidevice

        symbolic_gpu_id = outer_map.params[0]

        # Add the parameter of the outer map
        for node in graph.successors(inner_map_entry):
            if isinstance(node, nodes.NestedSDFG):
                map_syms = inner_map_entry.range.free_symbols
                for sym in map_syms:
                    symname = str(sym)
                    if symname not in node.symbol_mapping.keys():
                        node.symbol_mapping[symname] = sym
                        node.sdfg.symbols[symname] = graph.symbols_defined_at(
                            node)[symname]

        # Add transient Data leading to the inner map
        prefix = self.new_transient_prefix
        for node in graph.predecessors(outer_map_entry):
            # Only AccessNodes are relevant
            if (isinstance(node, nodes.AccessNode)
                    and not (self.skip_scalar
                             and isinstance(node.desc(sdfg), Scalar))):
                if self.use_p2p and node.desc(
                        sdfg).storage is dtypes.StorageType.GPU_Global:
                    continue

                in_data_node = InLocalStorage.apply_to(sdfg,
                                                       dict(array=node.data,
                                                            prefix=prefix),
                                                       verify=False,
                                                       save=False,
                                                       node_a=outer_map_entry,
                                                       node_b=inner_map_entry)
                in_data_node.desc(sdfg).location['gpu'] = symbolic_gpu_id
                in_data_node.desc(sdfg).storage = dtypes.StorageType.GPU_Global

        wcr_data: Dict[str, Any] = {}
        # Add transient Data leading to the outer map
        for edge in graph.in_edges(outer_map_exit):
            node = graph.memlet_path(edge)[-1].dst
            if isinstance(node, nodes.AccessNode):
                data_name = node.data
                # Transients with write-conflict resolution need to be
                # collected first as AccumulateTransient creates a nestedSDFG
                if edge.data.wcr is not None:
                    dtype = sdfg.arrays[data_name].dtype
                    redtype = operations.detect_reduction_type(edge.data.wcr)
                    # Custom reduction can not have an accumulate transient,
                    # as the accumulation from the transient to the outer
                    # storage is not defined.
                    if redtype == dtypes.ReductionType.Custom:
                        warnings.warn(
                            'Using custom reductions in a GPUMultitransformed '
                            'Map only works for a small data volume. For large '
                            'volume there is no guarantee.')
                        continue
                    identity = dtypes.reduction_identity(dtype, redtype)
                    wcr_data[data_name] = identity
                elif (not isinstance(node.desc(sdfg), Scalar)
                      or not self.skip_scalar):
                    if self.use_p2p and node.desc(
                            sdfg).storage is dtypes.StorageType.GPU_Global:
                        continue
                    # Transients without write-conflict resolution
                    if prefix + '_' + data_name in sdfg.arrays:
                        create_array = False
                    else:
                        create_array = True
                    out_data_node = OutLocalStorage.apply_to(
                        sdfg,
                        dict(array=data_name,
                             prefix=prefix,
                             create_array=create_array),
                        verify=False,
                        save=False,
                        node_a=inner_map_exit,
                        node_b=outer_map_exit)
                    out_data_node.desc(sdfg).location['gpu'] = symbolic_gpu_id
                    out_data_node.desc(
                        sdfg).storage = dtypes.StorageType.GPU_Global

        # Add Transients for write-conflict resolution
        if len(wcr_data) != 0:
            nsdfg = AccumulateTransient.apply_to(
                sdfg,
                options=dict(array_identity_dict=wcr_data, prefix=prefix),
                map_exit=inner_map_exit,
                outer_map_exit=outer_map_exit)
            nsdfg.schedule = dtypes.ScheduleType.GPU_Multidevice
            nsdfg.location['gpu'] = symbolic_gpu_id
            for transient_node in graph.successors(nsdfg):
                if isinstance(transient_node, nodes.AccessNode):
                    transient_node.desc(sdfg).location['gpu'] = symbolic_gpu_id
                    transient_node.desc(
                        sdfg).storage = dtypes.StorageType.GPU_Global
                    nsdfg.sdfg.arrays[
                        transient_node.label].location['gpu'] = symbolic_gpu_id
                    nsdfg.sdfg.arrays[
                        transient_node.
                        label].storage = dtypes.StorageType.GPU_Global
            infer_types.set_default_schedule_storage_types_and_location(
                nsdfg.sdfg, dtypes.ScheduleType.GPU_Multidevice,
                symbolic_gpu_id)

        # Remove the parameter of the outer_map from the sdfg symbols,
        # as it got added as a symbol in StripMining.
        if outer_map.params[0] in sdfg.free_symbols:
            sdfg.remove_symbol(outer_map.params[0])
