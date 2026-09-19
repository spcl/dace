import dace
import dace.sdfg.construction_utils as cutil


def _get_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG("sd1")
    s1 = sdfg.add_state("s1", is_start_block=True)

    sdfg.add_array("A", (5, 5), dace.float64)
    sdfg.add_array("B", (5, 5), dace.float64)
    sdfg.add_scalar("c", dace.float64)
    return sdfg, s1


def test_assignment_as_tasklet():
    sdfg, s1 = _get_sdfg()
    sdfg.validate()
    cutil.generate_assignment_as_tasklet_in_state(s1, "c", "A[4, 4] + 2.0 * B[1, 2]")
    sdfg.validate()


def test_read_modify_write_reads_the_in_connector():
    """``c = c >> 1`` names ``c`` on BOTH sides, and the read is an INPUT.

    The two sides are rewritten with separate maps for exactly this shape. Under one shared
    map the LHS entry overwrote the RHS one, so the body read the OUT connector -- which no
    edge feeds -- while the in connector it was given sat unused. Nothing rejected that: the
    name simply became a free symbol of the tasklet and surfaced much later, as a
    ``KeyError`` from ``SDFG.arglist`` over a symbol the SDFG never had.
    """
    sdfg, s1 = _get_sdfg()
    cutil.generate_assignment_as_tasklet_in_state(s1, "c", "c + 2.0")
    sdfg.validate()

    tasklet = next(n for n in s1.nodes() if isinstance(n, dace.nodes.Tasklet))
    in_conn = next(iter(tasklet.in_connectors))
    out_conn = next(iter(tasklet.out_connectors))
    lhs, _, rhs = tasklet.code.as_string.partition("=")

    assert in_conn in rhs, f"the read of `c` is an input, but the body reads {rhs.strip()!r}"
    assert out_conn not in rhs, f"the body reads its own out connector: {rhs.strip()!r}"
    assert out_conn in lhs, f"the write must target the out connector, got {lhs.strip()!r}"
    # Every name the body reads is fed by an edge, so none of them escapes as a symbol.
    assert not ({str(s) for s in tasklet.free_symbols} - set(tasklet.in_connectors)), \
        f"tasklet reads names nothing feeds: {tasklet.free_symbols}"


if __name__ == "__main__":
    test_assignment_as_tasklet()
    test_read_modify_write_reads_the_in_connector()
