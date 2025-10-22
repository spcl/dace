import dace
import dace.sdfg.construction_utils as cutil

def _get_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG("sd1")
    s1 = sdfg.add_state("s1", is_start_block=True)

    sdfg.add_array(
        "A", (5,5), dace.float64
    )
    sdfg.add_array(
        "B", (5,5), dace.float64
    )
    sdfg.add_scalar(
        "c", dace.float64
    )
    return sdfg, s1

def test_assignment_as_tasklet():
    sdfg, s1 = _get_sdfg()
    sdfg.validate()
    cutil.generate_assignment_as_tasklet_in_state(s1, "c", "A[4, 4] + 2.0 * B[1, 2]")
    sdfg.validate()

if __name__ == "__main__":
    test_assignment_as_tasklet()