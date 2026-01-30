# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from itertools import chain
from pathlib import Path
from typing import Generator


def find_all_f90_files(root: Path) -> Generator[Path, None, None]:
    if root.is_file():
        yield root
    else:
        for f in chain(root.rglob("*.f90"), root.rglob("*.F90"), root.rglob("*.incf")):
            yield f
