"""Put the HLFIR test directory on sys.path so ``from _util import ...`` works
when pytest collects tests from this folder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
