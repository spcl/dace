# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.sdfg.sdfg import SDFG, InterstateEdge, LogicalGroup

from dace.sdfg.state import SDFGState, ControlFlowBlock, ControlFlowGraph, ScopeBlock, LoopScopeBlock, BranchScopeBlock

from dace.sdfg.scope import (scope_contains_scope, is_devicelevel_gpu, devicelevel_block_size, ScopeSubgraphView)

from dace.sdfg.replace import replace, replace_properties, replace_properties_dict

from dace.sdfg.utils import (has_dynamic_map_inputs, dynamic_map_inputs, is_parallel, concurrent_subgraphs,
                             find_input_arraynode, find_output_arraynode, trace_nested_access, is_array_stream_view,
                             local_transients, load_precompiled_sdfg)

from dace.sdfg.validation import (InvalidSDFGError, InvalidSDFGNodeError, InvalidSDFGEdgeError,
                                  InvalidSDFGInterstateEdgeError, NodeNotExpandedError)
