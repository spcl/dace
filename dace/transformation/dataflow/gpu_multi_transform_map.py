# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains the GPUMultiTransformMap transformation. """

from dace import dtypes, registry
from dace.data import Scalar
from dace.sdfg import has_dynamic_map_inputs
from dace.sdfg import utils as sdutil
from dace.sdfg import nodes, SDFG, SDFGState, infer_types
from dace.properties import make_properties, Property, SymbolicProperty
from dace.transformation import transformation
from dace.properties import make_properties
from dace.config import Config
from dace.symbolic import SymbolicType
from typing import Dict, Any


@registry.autoregister_params(singlestate=True)
@make_properties
class GPUMultiTransformMap(transformation.Transformation):
    """ Implements the GPUMultiTransformMap transformation.

        Tiles a single map into 2 maps. The outer map is of schedule 
        GPU_Multidevice and loops over the gpus, while the inner map
        is a GPU-scheduled map. It also creates GPU transient arrays
        between the two maps.
    """
    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    new_dim_prefix = Property(dtype=str,
                              default="gpu",
                              allow_none=True,
                              desc="Prefix for new dimension name")
    new_transient_prefix = Property(dtype=str,
                                    default="gpu_multi_",
                                    allow_none=True,
                                    desc="Prefix for the transient name")
    skip_scalar = Property(
        dtype=bool,
        default=True,
        allow_none=True,
        desc="If True: skips the scalar data nodes. "
        "If False: creates localstorage for scalar transients.")
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
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_entry = graph.nodes()[candidate[GPUMultiTransformMap._map_entry]]

        # Check if there is more than one GPU available:
        if (Config.get("compiler", "cuda", "max_number_gpus") < 2):
            return False

        # Check if the map is one-dimensional
        if map_entry.map.range.dims() != 1:
            return False

        # We cannot transform a map which is already on a GPU or FPGA
        if map_entry.map.schedule in dtypes.GPU_SCHEDULES + [
                dtypes.ScheduleType.GPU_Default, dtypes.ScheduleType.FPGA_Device
        ]:
            return False

        # We cannot transform a map which is already inside a GPU map, or on
        # another device
        schedule_whitelist = [
            dtypes.ScheduleType.Default, dtypes.ScheduleType.Sequential
        ]
        sdict = graph.scope_dict()
        parent = sdict[map_entry]
        while parent is not None:
            if parent.map.schedule not in schedule_whitelist:
                return False
            parent = sdict[parent]

        # Dynamic map ranges not supported
        if has_dynamic_map_inputs(graph, map_entry):
            return False

        # # Only one WCR is currently supported, limited by AccumulateTransient
        # map_exit = graph.exit_node(map_entry)
        # if sum(1 for e in graph.out_edges(map_exit) if e.data.wcr) > 1:
        #     return False

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
        from dace.frontend import operations

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

        # Add transient Data leading to the inner map
        prefix = self.new_transient_prefix
        for node in graph.predecessors(outer_map_entry):
            # Skip scalar data
            if isinstance(node, nodes.AccessNode):
                if self.skip_scalar and isinstance(node.desc(sdfg), Scalar):
                    continue
                in_data_node = InLocalStorage.apply_to(sdfg,
                                                       dict(array=node.data,
                                                            prefix=prefix),
                                                       verify=False,
                                                       save=False,
                                                       node_a=outer_map_entry,
                                                       node_b=inner_map_entry)

        for node in graph.successors(inner_map_entry):
            if isinstance(node, nodes.NestedSDFG):
                map_syms = inner_map_entry.range.free_symbols
                for sym in map_syms:
                    symname = str(sym)
                    if symname not in node.symbol_mapping.keys():
                        node.symbol_mapping[symname] = sym
                        node.sdfg.symbols[symname] = graph.symbols_defined_at(
                            node)[symname]
        wcr_data: Dict[str, Any] = {}
        # Add transient Data leading to the outer map
        for edge in graph.out_edges(outer_map_exit):
            node = edge.dst
            # skip scalar data
            if isinstance(node, nodes.AccessNode):
                data_name = node.data
                # Transients with write-conflict resolution need to be
                # collected first as AccumulateTransient creates a nestedSDFG
                if edge.data.wcr is not None:
                    dtype = sdfg.arrays[data_name].dtype
                    redtype = operations.detect_reduction_type(edge.data.wcr)
                    identity = dtypes.reduction_identity(dtype, redtype)
                    wcr_data[data_name] = identity
                elif not isinstance(node.desc(sdfg), Scalar):
                    # Transients without write-conflict resolution
                    if prefix + data_name in sdfg.arrays:
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

        if len(wcr_data) != 0:
            nsdfg = AccumulateTransient.apply_to(
                sdfg,
                options=dict(array_identity_dict=wcr_data, prefix=prefix),
                map_exit=inner_map_exit,
                outer_map_exit=outer_map_exit)

        # Propagate schedule, storage and location
        infer_types.set_default_schedule_storage_types_and_location(sdfg, None)

        # Remove the parameter of the outer_map from the sdfg symbols,
        # as it got added as a symbol in StripMining.
        if outer_map.params[0] in sdfg.free_symbols:
            sdfg.remove_symbol(outer_map.params[0])
