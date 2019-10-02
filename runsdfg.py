import dace
import dace.graph.labeling
import sys
import time

print(time.time(), 'loading')
a = dace.SDFG.from_file(sys.argv[1])
print(time.time(), 'propagating')
dace.graph.labeling.propagate_labels_sdfg(a)
print(time.time(), 'drawing')
a.draw_to_file()
exit()

a.apply_strict_transformations()

a.apply_strict_transformations()

a.draw_to_file()
