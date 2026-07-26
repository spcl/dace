import sys
from pathlib import Path
sys.path.insert(0, '/home/primrose/Work/dace-fortran')
import importlib.util
from dace.sdfg import SDFG
PIPE = Path('/home/primrose/Work/dace/tests/corpus/cloudsc/pipelines.py')
spec = importlib.util.spec_from_file_location('p', PIPE); pl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pl)
g = SDFG.from_file(sys.argv[1])
def selfassigns():
    r=[]
    for sd in g.all_sdfgs_recursive():
        for e in sd.all_interstate_edges():
            for k,v in e.data.assignments.items():
                if str(k)==str(v):
                    r.append((str(k), f"{e.src}->{e.dst}"))
    return r
pre=selfassigns()
print(f"PRE-finalize self-assigns: {len(pre)} -> {pre[:8]}")
fin = [s for nm, s in pl.variant_phases('canon_cpu', constants={}) if nm == 'finalize'][0]
prev=len(pre)
for i,(label,unit) in enumerate(fin):
    unit(g)
    n=selfassigns()
    if len(n)!=prev:
        print(f"[{i}] {label}: self-assigns {prev} -> {len(n)}  sample={n[:5]}")
        prev=len(n)
    if i>=18: break  # stop after fuse group
