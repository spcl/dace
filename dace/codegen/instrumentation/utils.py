from dace import dtypes, nodes as dace_nodes
from dace.sdfg.sdfg import SDFG


def instrument_sdfg(instrumentation_type: dtypes.InstrumentationType, sdfg: SDFG) -> None:
    """
    Instrument everything on the given SDFG with the specified instrumentation type.

    :param instrumentation: The instrumentation type to apply.
    :param sdfg: The SDFG to instrument.
    """
    sdfg.instrument = instrumentation_type
    for state in sdfg.states():
        state.instrument = instrumentation_type
        for node in state.nodes():
            if isinstance(node, dace_nodes.MapEntry):
                node.map.instrument = instrumentation_type
            elif isinstance(node, dace_nodes.NestedSDFG):
                instrument_sdfg(instrumentation_type, node.sdfg)
