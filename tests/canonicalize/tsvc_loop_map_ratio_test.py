# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""For-loop vs map ratio after ``canonicalize``, per TSVC benchmark group.

Builds the simplified SDFG of every kernel in the ``tsvc_2`` corpus
(:mod:`tests.corpus.tsvc`) and the ``tsvc_2_5`` extension corpus
(:mod:`tests.corpus.tsvc_2_5`), runs the production ``canonicalize`` recipe, and
counts sequential ``LoopRegion`` s vs parallel ``MapEntry`` s. ``loop_map_table``
returns a list of markdown rows (header + separator + one row per group), so the
result renders directly as a markdown table.
"""
import contextlib
import io
import os

# dace's frontend lazily ``from mpi4py import MPI`` during ``to_sdfg``; on hosts
# where MPI_Init stalls on UCX, steer Open MPI off UCX before that import runs.
# ``setdefault`` keeps any externally-provided MPI configuration.
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

from dace.sdfg import nodes as nd
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc_2_5 import tsvc_2_5

#: Corpus defaults, matching tsvc_corpus_test / measure_parallelization. The
#: unroll cap keeps ``heat3d``'s constant tile-8 inner loops from unrolling 512x
#: (otherwise a single kernel runs for minutes); the rest is the production recipe.
_PEEL_LIMIT = 4
_BREAK_ANTI_DEP = True
_UNROLL_LIMIT = 4

#: (label, list of @dace.program objects).
_GROUPS = [
    ("tsvc_2", [k.program for k in tsvc.collect()]),
    ("tsvc_2_5", tsvc_2_5.collect()),
]


def _count(program):
    """Canonicalize the program's simplified SDFG; return ``(for_loops, maps)``."""
    sdfg = program.to_sdfg(simplify=True)
    with contextlib.redirect_stdout(io.StringIO()):
        canonicalize(sdfg, peel_limit=_PEEL_LIMIT, break_anti_dependence=_BREAK_ANTI_DEP, unroll_limit=_UNROLL_LIMIT)
    loops = sum(1 for cfr in sdfg.all_control_flow_regions() if isinstance(cfr, LoopRegion))
    maps = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry))
    return loops, maps


def loop_map_table():
    """:returns: markdown table rows (``list[str]``) of for-loop% per benchmark group."""
    rows = ["| benchmark | kernels | maps | for-loops | for-loop % |", "| --- | ---: | ---: | ---: | ---: |"]
    for label, programs in _GROUPS:
        loops = maps = 0
        for p in programs:
            l, m = _count(p)
            loops += l
            maps += m
        total = loops + maps
        pct = f"{100.0 * loops / total:.1f}%" if total else "n/a"
        rows.append(f"| {label} | {len(programs)} | {maps} | {loops} | {pct} |")
    return rows


def test_loop_map_ratio_table():
    """``canonicalize`` over both corpora produces a renderable for-loop% table."""
    rows = loop_map_table()
    print("\n".join(rows))
    assert len(rows) == 2 + len(_GROUPS)  # header + separator + one row per group


if __name__ == "__main__":
    print("\n".join(loop_map_table()))
