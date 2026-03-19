#!/usr/bin/env python3
import re
from pathlib import Path
from collections import Counter

# File to scan
FILE = "tests/passes/tsvc_vectorization_test.py"

# Regex that extracts the pattern test_s1234
test_re = re.compile(r"def\s+test_(s\d+)\s*\(")

text = Path(FILE).read_text()

# Count occurrences
counter = Counter(test_re.findall(text))

# Report duplicates
duplicates = {name: count for name, count in counter.items() if count > 1}

if duplicates:
    print("Duplicate test definitions detected:\n")
    for name, count in sorted(duplicates.items()):
        print(f"  {name}: {count} occurrences")
else:
    print("No duplicate test_s<int> definitions found ✔")
