#!/usr/bin/env python3
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# CONFIG — which directory to scan
# ---------------------------------------------------------------------
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

# Regex patterns
kernel_re = re.compile(r"def\s+(dace_s\d+)\s*\(")
test_re = re.compile(r"def\s+(test_s\d+)\s*\(")

# Storage
kernels = set()
tests = set()

# ---------------------------------------------------------------------
# Scan all .py files recursively
# ---------------------------------------------------------------------
p = Path("tests/passes/tsvc_vectorization_test.py")
text = p.read_text()

for m in kernel_re.finditer(text):
    kernels.add(m.group(1))

for m in test_re.finditer(text):
    # test_sXXXX → sXXXX
    fname = m.group(1)
    tests.add(fname.replace("test_", "dace_"))

# ---------------------------------------------------------------------
# Report missing tests
# ---------------------------------------------------------------------
missing = sorted(kernels - tests)

print("\n=== Kernels missing test functions ===")
if not missing:
    print("✔ All kernels have corresponding test_s<kernel>()")
else:
    for k in missing:
        print("  -", k)

print("\nTotal kernels:", len(kernels))
print("Total tests:", len(tests))
print("Missing:", len(missing))
