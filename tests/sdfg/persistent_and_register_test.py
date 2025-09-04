import dace
import numpy

def _get_sdfg():
    sdfg: dace.SDFG = dace.SDFG("test_persistent_lifetime_and_register_storage")
    _, perA = sdfg.add_array(
        name="perA",
        shape=[10],
        dtype=dace.float64,
        storage=dace.dtypes.StorageType.CPU_Heap,
        transient=True,
        lifetime=dace.dtypes.AllocationLifetime.Persistent,
        find_new_name=False
    )
    _, inA = sdfg.add_array(
        name="inA",
        shape=[10],
        dtype=dace.float64,
        storage=dace.dtypes.StorageType.CPU_Heap,
        transient=False,
        find_new_name=False
    )
    state = sdfg.add_state("main_state")
    an_perA = state.add_access("perA")
    an_inA = state.add_access("inA")
    state.add_edge(
        an_perA,
        None,
        an_inA,
        None,
        dace.memlet.Memlet.from_array(an_perA, perA)
    )
    return sdfg

def test_persistent_lifetime_and_register_storage():
    sdfg = _get_sdfg()
    sdfg.validate()
    sdfg.save("s.sdfgz", compress=True)
    csdfg = sdfg.compile()
    inA = numpy.zeros((10,))
    print(inA)
    csdfg(inA=inA)
    print(inA)


if __name__ == "__main__":
    test_persistent_lifetime_and_register_storage()