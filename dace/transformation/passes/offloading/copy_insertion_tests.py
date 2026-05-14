import dace
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState, ConditionalBlock, ControlFlowRegion, LoopRegion, ReturnBlock, ContinueBlock, BreakBlock, ControlFlowBlock
from dace import dtypes
from dace.transformation.passes.offloading.OffloadToAccelerator import OffloadToAccelerator as OtA


sdfg = ()
#sdfg.view()

IR = OtA().get_IR(sdfg)
print(IR)

# TODO:
# look through previous testcases, see what still applies
# collect in automatic test suite
# add new test cases to suite
#   simple for all 4 scenarios
#   more involved for all 4 scenarios
# before you fix the bugs:
#   add yakups old test cases to the suite
#   add big sdfgs to the suite
#   add heat3d & npbench to the suite
# 1: have a fully functional suite, don't matter if some test don't pass yet
# 2: get all test cases to run
# 3: organize git, grant access to yakup, open overleaf doc, read example thesis, begin writing


# TODO: scalars have pass by copy right now -> if map writes to single scalar, detect and raise error or convert to array of length one and properly offload
# curently not offloaded -> incorrect -> run "replace_all_length1_arrays_with_scalars" at start, then replace back iff written to by GPU / map

# todo(?): make interstate edge copy method also work with controlflowblocks 

# TODO: dace rep -> tests -> npbench -> polybench -> copy2d, copy1d, heat3d, ... -> lib nodes, wcr edges etc. -> use as test cases
# views, subset and wcr edges might create problems -> if they do, discuss with Yakup, might not need to handle this
# use npbench -> polybench and s-cases

# TODO: clean up -> PR branch -> write thesis!

"""
thesis
introduction: rephrase thesis globals
list contributions
related work: how do others lower things to GPU
implementation: document code and ask claude to summarize
evaluation: test against nbench, test sdfgs which failed previously
"""