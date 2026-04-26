"""Put the parent ``tests/hlfir`` directory on sys.path so ``from _util
import ...`` resolves the same way the FaCe-native tests do."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
