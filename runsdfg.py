import dace

import sys

a = dace.SDFG.from_file(sys.argv[1])
a.draw_to_file()
exit()

a.apply_strict_transformations()

a.apply_strict_transformations()

a.draw_to_file()
