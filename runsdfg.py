# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import sys
import time

print(time.time(), 'loading')
a = dace.SDFG.from_file(sys.argv[1])
print(time.time(), 'propagating')
dace.propagate_memlets_sdfg(a)
print(time.time(), 'strict transformations')
a.apply_strict_transformations()
print(time.time(), 'saving')
a.save('strict.sdfg')
print(time.time(), 'compiling')
a.compile()
