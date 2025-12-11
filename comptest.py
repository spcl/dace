import dace

sdfg = dace.SDFG.from_file("x0.sdfg")
sdfg.validate()

sdfg.compile()
