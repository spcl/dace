import dace
import sys
from pathlib import Path

print("dace path:", list(dace.__path__))

workspace_root = Path(__file__).resolve().parents[5]
npbench_root = workspace_root / "npbench"
if npbench_root.is_dir() and str(npbench_root) not in sys.path:
	sys.path.insert(0, str(npbench_root))

from run_framework import run_benchmark

result = run_benchmark(benchname="heat_3d", fname="dace_gpu", preset="S", validate=True, repeat=1, timeout=100, ignore_errors=False, save_strict=False, load_strict=False)
print(result)