import dace

@dace.program
def _p(A: dace.float64[10]):
    for i in dace.map[0:10:5]:
        A[i] = A[i] * 2

def test_yield():
    # It would previously crash
    sdfg = _p.to_sdfg()
    sdfg.name = "_P"
    sdfg.save("x.sdfg")
    for s in sdfg.all_states():
        for n in s.nodes():
            if isinstance(n, dace.nodes.MapEntry):
                nl = [p for p in s.all_simple_paths(n, s.exit_node(n), False)]
                assert len(nl) == 1
                assert len(nl[0]) == 5
                
            if isinstance(n, dace.nodes.MapEntry):
                nl = [p for p in s.all_simple_paths(n, s.exit_node(n), True)]
                assert len(nl) == 1
                assert len(nl[0]) == 4


if __name__ == "__main__":
    test_yield()