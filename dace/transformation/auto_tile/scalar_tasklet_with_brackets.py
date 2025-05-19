import dace

sdfg = dace.SDFG("test")

s1 = sdfg.add_state("state")
sdfg.add_array("A", shape=(1,), dtype=dace.float32)

an = s1.add_access("A")
s1.add_tasklet(
    name="t1",
    inputs={"_in_a"},
    outputs={"_out_a"},
    code="{ _out_a = _in_a + 1; }",
    language=dace.dtypes.Language.CPP,
)
an2 = s1.add_access("A")

s1.add_edge(an, "_in_a", "t1", "_in_a", dace.Memlet("A[0]"))
s1.add_edge("t1", "_out_a", an2, "_out_a", dace.Memlet("A[0]"))

sdfg.save("scalar_tasklet_with_brackets.sdfg")

