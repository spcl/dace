# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lower an SDFG to the form the Python emitter takes."""

from dace import SDFG, dtypes
from dace.codegen.exceptions import CodegenError
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.sdfg import infer_types, nodes
from dace.sdfg.state import ControlFlowRegion
from dace.transformation.pass_pipeline import FixedPointPipeline
from dace.transformation.passes.inline_tasklet_connectors import InlineTaskletConnectors
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising
from dace.transformation.passes.unique_names import UniqueNames


def require_structured(sdfg: SDFG) -> None:
    """Raise unless every region of ``sdfg`` is a chain of blocks joined by unconditional edges."""
    for region in sdfg.all_control_flow_regions(recursive=True):
        if not isinstance(region, ControlFlowRegion):
            continue
        for block in region.nodes():
            out_edges = region.out_edges(block)
            if len(out_edges) > 1 or any(not edge.data.is_unconditional() for edge in out_edges):
                raise CodegenError(
                    f"Python backend: block {block.label!r} of {region.label!r} branches on an interstate edge; "
                    "it needs structured control flow (ControlFlowRaising)"
                )


def require_python_tasklets(sdfg: SDFG) -> None:
    """Raise on any node the emitter has no Python form for."""
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.Tasklet) and node.language is not dtypes.Language.Python:
            raise CodegenError(f"Python backend: tasklet {node.label!r} is {node.language.name}, not Python")
        if isinstance(node, nodes.LibraryNode) and not isinstance(node, TileOp):
            raise CodegenError(f"Python backend: library node {node.label!r} has no Python expansion")


def expand_library_nodes(sdfg: SDFG) -> None:
    """Expand every library node, one level at a time, by its ``CuPy`` expansion where it has one."""
    # Avoid import loop
    from dace.libraries.python_expansions import IMPLEMENTATION

    while True:
        # A tile node is a statement of its tile kernel, never expanded
        pending = [
            (n, s)
            for n, s in sdfg.all_nodes_recursive()
            if isinstance(n, nodes.LibraryNode) and not isinstance(n, TileOp)
        ]
        if not pending:
            return
        for node, state in pending:
            if IMPLEMENTATION in node.implementations:
                node.implementation = IMPLEMENTATION
            node.expand(state)


def prepare(sdfg: SDFG, validate: bool = False) -> None:
    """Expand, raise, flatten and rename ``sdfg`` in place so every name of it lives in one Python namespace.

    :param validate: Validate the prepared SDFG once at the end.
    """
    # Avoid import loop
    from dace.codegen.codegen import inline_host_nested_sdfgs

    expand_library_nodes(sdfg)
    FixedPointPipeline([ControlFlowRaising()]).apply_pass(sdfg, {})
    infer_types.infer_connector_types(sdfg)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    require_structured(sdfg)
    require_python_tasklets(sdfg)
    inline_host_nested_sdfgs(sdfg, validate=False)
    InlineTaskletConnectors().apply_pass(sdfg, {})
    UniqueNames().apply_pass(sdfg, {})
    if validate:
        sdfg.validate()
