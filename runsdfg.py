import dace
import dace.graph.labeling
import sys
import time

print(time.time(), 'loading')
a = dace.SDFG.from_file(sys.argv[1])
print(time.time(), 'propagating')
dace.graph.labeling.propagate_labels_sdfg(a)
print(time.time(), 'strict transformations')
a.apply_strict_transformations()
print(time.time(), 'saving')
a.save('strict.sdfg')
print(time.time(), 'compiling')
a.compile(optimizer=False)
