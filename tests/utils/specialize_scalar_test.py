import dace
import pytest
import dace.sdfg.utils as sdutil

def _get_sdfg_for_dynamic_map_input():
    sdfg = dace.SDFG("dynamic_input")
    sdfg.add_scalar("nlev", dtype=dace.int32, transient=False)
    sdfg.add_symbol("nlev_sym", stype=dace.int32)
    s0 = sdfg.add_state("s0")
    sdfg.add_array("A", ["nlev_sym"], dtype=dace.float64, transient=False)
    s1 = sdfg.add_state("s1")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"nlev_sym": "nlev"}))
    an = s1.add_access("A")
    _, _, _ = s1.add_mapped_tasklet(
        name="assign_map",
        map_ranges={"i": dace.subsets.Range(ranges=[("0","nlev-1","1"),])},
        inputs={},
        outputs={"_out": dace.memlet.Memlet("A[i]")},
        code="_out = 0",
        input_nodes=None,
        external_edges=True,
        output_nodes={"A": an}
    )
    return sdfg


def _get_sdfg_with_symbol_use_in_if():
    pass

def test_specialize_with_dynamic_input():
    sdfg = _get_sdfg_for_dynamic_map_input()
    sdfg.validate()
    sdutil.specialize_scalar(sdfg=sdfg, scalar_name="nlev", scalar_val="90")
    sdfg.validate()
    sdfg.compile()
    map_entries = set()
    for s in sdfg.all_states():
        for n in s.nodes():
            if isinstance(n, dace.nodes.MapEntry):
                map_entries.add(n)
    assert len(map_entries) == 1
    map_entry: dace.nodes.MapEntry = next(iter(map_entries))
    range: dace.subsets.Range = map_entry.map.range
    assert range == dace.subsets.Range(ranges=[("0", "89", "1")])

if __name__ == "__main__":
    test_specialize_with_dynamic_input()

