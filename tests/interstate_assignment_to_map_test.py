from typing import Dict
import dace

from dace.transformation.passes.interstate_assignment_to_map import InterstateAssignmentToMap

N = dace.symbol("N")

def _get_interstate_dependent_sdfg(assignments: Dict, symbols_at_start=False):
    sdfg = dace.SDFG("interstate_dependent")
    for k in assignments:
        sdfg.add_symbol(k, dace.int32)

    s1 = sdfg.add_state("s1")
    s2 = sdfg.add_state("s2")

    if not symbols_at_start:
        s0 = sdfg.add_state("s0")
        pre_assignments = dict()
        for k,v in assignments.items():
            pre_assignments[k] = v*2
        sdfg.add_edge(s0, s1, dace.InterstateEdge(None, assignments=pre_assignments))

    for sid, s in [("1", s1), ("2", s2)]:
        sdfg.add_array(f"array{sid}", (N, ) , dace.int32, storage=dace.StorageType.CPU_Heap, transient=True)
        an = s.add_access(f"array{sid}")
        an2 = s.add_access(f"array{sid}")
        t = s.add_tasklet(f"tasklet{sid}", {"_in"}, {"_out"}, "_out = _in * 2")
        map_entry, map_exit = s.add_map(f"map{sid}", {"i":dace.subsets.Range([(0,N-1,1)])})
        for m in [map_entry, map_exit]:
            m.add_in_connector(f"IN_array{sid}")
            m.add_out_connector(f"OUT_array{sid}")
        s.add_edge(an, None, map_entry, f"IN_array{sid}", dace.memlet.Memlet(f"array{sid}[0:N]"))
        s.add_edge(map_entry, f"OUT_array{sid}", t, "_in", dace.memlet.Memlet(f"array{sid}[i]"))
        s.add_edge(t, "_out", map_exit, f"IN_array{sid}", dace.memlet.Memlet(f"array{sid}[i]"))
        s.add_edge(map_exit, f"OUT_array{sid}", an2, None, dace.memlet.Memlet(f"array{sid}[0:N]"))

    sdfg.add_edge(s1, s2, dace.InterstateEdge(None, assignments=assignments))
    sdfg.save("s1.sdfg")
    sdfg.validate()
    return sdfg

def test_interstate_assignment():
    sdfg = _get_interstate_dependent_sdfg({"N": 5}, False)
    sdfg.validate()
    sdfg()
    iatm =  InterstateAssignmentToMap()
    iatm.apply_pass(sdfg, {})
    sdfg.save("s2.sdfg")
    sdfg.validate()
    sdfg()

if __name__ == "__main__":
    test_interstate_assignment()