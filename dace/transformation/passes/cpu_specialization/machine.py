# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""What the host CPU actually is, and the thresholds that follow from it.

The CPU specialization band decides what stays sequential. Canonical form is parallel -- that is not
in question here -- but a parallel region has to be paid for, and whether it pays depends on the
machine the code will run on, not on the machine the pass was written on. Two thresholds govern it,
and both were hardcoded constants measured on ONE development box:

* ``compiler.cpu.parallel_min_work_per_region`` -- iterations a top-level map needs to earn its own
  fork/join.
* ``compiler.cpu.parallel_transfer_min_elements`` -- elements a bulk copy or fill needs before an
  OpenMP element map beats a single ``memcpy``/``memset``.

Both are the SAME question: when does the work amortize a fork/join? Sequential time is
``N * t``; parallel is ``fork(P) + N * t / P``; they cross at ``N = fork(P) / (t * (1 - 1/P))``,
and ``fork(P)`` grows roughly linearly in the team size. So the break-even count grows with the
core count -- a 72-core machine needs MORE work per region than an 8-core one, not less, even
though it finishes that work faster.

That is the whole model. The absolute scale comes from one measurement (see
:data:`REFERENCE_CORES`), and the topology comes from the machine through sysfs and ``sysconf``, so
moving to another CPU rescales the thresholds instead of inheriting the development box's numbers.
"""
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

#: Physical cores of the machine the reference numbers below were measured on (AMD Ryzen 7 8845HS,
#: 8 cores / 16 threads). An empty ``omp parallel if(0)`` region measured 691 ns there at 8 threads,
#: against the 90 ns a 137-element scan takes -- the fork/join that these thresholds exist to
#: amortize.
REFERENCE_CORES = 8

#: Reference values at :data:`REFERENCE_CORES`, i.e. the constants these thresholds used to be.
REFERENCE_MIN_WORK_PER_REGION = 256
REFERENCE_TRANSFER_MIN_ELEMENTS = 262144

CPU_ROOT = Path('/sys/devices/system/cpu')
NODE_ROOT = Path('/sys/devices/system/node')


@dataclass(frozen=True, slots=True)
class CpuTopology:
    """The host's shape, as far as the specialization band needs it.

    :ivar physical_cores: cores that do not share execution resources; the unit a parallel region
                          actually scales on. SMT siblings are NOT counted -- two hyperthreads share
                          one core's memory pipeline, so a bandwidth-bound map gains nothing from
                          the second.
    :ivar logical_cores: what the OS schedules on, SMT siblings included.
    :ivar l1d_bytes: L1 data cache of one core.
    :ivar l2_bytes: L2 of one core (or of the cluster that shares it).
    :ivar llc_bytes: last level cache, usually shared by the whole package.
    :ivar numa_nodes: NUMA domains; more than one means a first-touch policy decides bandwidth.
    """
    physical_cores: int
    logical_cores: int
    l1d_bytes: int
    l2_bytes: int
    llc_bytes: int
    numa_nodes: int


def read_int(path: Path) -> Optional[int]:
    """First integer in ``path``, or ``None`` when it cannot be read."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def read_size(path: Path) -> Optional[int]:
    """A sysfs cache size (``32K``, ``1024K``, ``16M``) in bytes, or ``None``."""
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    scale = {'K': 1024, 'M': 1024**2, 'G': 1024**3}.get(raw[-1:].upper())
    try:
        return int(raw[:-1]) * scale if scale else int(raw)
    except ValueError:
        return None


def count_physical_cores() -> Optional[int]:
    """Distinct physical cores, by grouping the SMT siblings sysfs reports for each CPU."""
    groups = set()
    for topo in sorted(CPU_ROOT.glob('cpu[0-9]*/topology')):
        for name in ('core_cpus_list', 'thread_siblings_list'):
            try:
                groups.add((topo / name).read_text().strip())
                break
            except OSError:
                continue
    return len(groups) or None


def cache_sizes() -> tuple:
    """``(l1d, l2, llc)`` in bytes from cpu0's cache indices; zeros for levels sysfs does not show."""
    l1d = l2 = llc = 0
    for index in sorted(CPU_ROOT.glob('cpu0/cache/index[0-9]*')):
        level, size = read_int(index / 'level'), read_size(index / 'size')
        if level is None or not size:
            continue
        try:
            kind = (index / 'type').read_text().strip()
        except OSError:
            kind = 'Unified'
        if level == 1 and kind in ('Data', 'Unified'):
            l1d = max(l1d, size)
        elif level == 2:
            l2 = max(l2, size)
        elif level >= 3:
            llc = max(llc, size)
    return l1d, l2, llc


@lru_cache(maxsize=1, typed=True)
def topology() -> CpuTopology:
    """The host topology, probed once.

    Every field degrades to a usable default rather than raising: this runs inside a compilation
    pipeline, and a machine whose sysfs is absent (a container, a non-Linux host) must still
    compile -- it just gets the reference machine's thresholds.
    """
    logical = os.cpu_count() or REFERENCE_CORES
    physical = count_physical_cores() or logical
    l1d, l2, llc = cache_sizes()
    nodes = len(list(NODE_ROOT.glob('node[0-9]*'))) or 1
    return CpuTopology(physical_cores=max(1, physical),
                       logical_cores=max(1, logical),
                       l1d_bytes=l1d or 32 * 1024,
                       l2_bytes=l2 or 1024 * 1024,
                       llc_bytes=llc or 16 * 1024**2,
                       numa_nodes=nodes)


def scale_for_team(reference: int, cores: Optional[int] = None) -> int:
    """``reference``, calibrated at :data:`REFERENCE_CORES`, rescaled to this machine's core count.

    Linear in the core count because that is how the fork/join it amortizes grows. A bigger machine
    therefore demands MORE work before it opens a region, which is not a contradiction: it finishes
    that work faster once it does.
    """
    physical = cores if cores is not None else topology().physical_cores
    return max(1, round(reference * physical / REFERENCE_CORES))


def min_work_per_region() -> int:
    """Iterations a top-level map needs before its own OpenMP region pays for itself."""
    return scale_for_team(REFERENCE_MIN_WORK_PER_REGION)


def transfer_min_elements() -> int:
    """Elements a bulk copy or fill needs before an OpenMP element map beats one libc call."""
    return scale_for_team(REFERENCE_TRANSFER_MIN_ELEMENTS)
