import dace
import json

from dace.optimization.superoptimization.superoptimizer import Superoptimizer


sdfg = dace.SDFG.from_file("samples/optimization/clusters/cluster_3/go_fast_S/maps_0/maps_0.sdfg")

optimizer = Superoptimizer(sdfg)
dreport = optimizer.dry_run(sdfg)
tuned = optimizer.tune(apply=False, compile_folder="/home/lukas/Documents/repos/autodace/ramdisk")

with open("tuned.sdfg", "w") as handle:
    json.dump(tuned.to_json(), handle)
