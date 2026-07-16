# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Theoretical peak memory bandwidth of the host DRAM and the GPU.

The LogP/LogGP memory model needs a DENOMINATOR: the achievable bandwidth a benchmark measures only
means something next to the hardware peak (achievable is typically 50-85% of it). This module reports
that peak from hardware facts, never from a guess.

Host DRAM comes from ``dmidecode --type 17``, which needs root: either run it live via
:func:`run_dmidecode` (sudo will prompt) or read a dump saved once with
``sudo dmidecode -t 17 > dram.txt``. :func:`host_dram_spec` picks between the two. Peak is the sum
over POPULATED devices of ``speed x data_width``, which is correct for both DIMMs (e.g. 2 x 64-bit
DDR5-5600 -> 89.6 GB/s) and soldered LPDDR (e.g. 4 x 32-bit LPDDR5x-6400 -> 102.4 GB/s); the usual
per-channel-count formula reports 2x that for the soldered case.

GPU: read from the CUDA device properties, so no file is needed --
``memory_clock x 2 (DDR) x bus_width / 8``.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

BITS_PER_BYTE = 8
MT_PER_S = 1e6


@dataclass(frozen=True)
class MemoryDevice:
    """One populated memory device (a DIMM, or a soldered LPDDR package) from a dmidecode dump."""
    size_bytes: int
    speed_mtps: float
    data_width_bits: int
    memory_type: str

    def peak_bytes_per_s(self) -> float:
        """Peak transfer rate of this device: transfers/s x bytes/transfer. The DDR double-pumping is
        already inside the MT/s figure dmidecode reports, so it is NOT doubled again here."""
        return self.speed_mtps * MT_PER_S * self.data_width_bits / BITS_PER_BYTE


@dataclass(frozen=True)
class DramSpec:
    """The host DRAM configuration, as read from a dmidecode dump."""
    devices: List[MemoryDevice]

    @property
    def memory_type(self) -> str:
        return self.devices[0].memory_type if self.devices else "unknown"

    @property
    def total_width_bits(self) -> int:
        return sum(d.data_width_bits for d in self.devices)

    def peak_bytes_per_s(self) -> float:
        """Theoretical peak host memory bandwidth: the sum over populated devices."""
        return sum(d.peak_bytes_per_s() for d in self.devices)


def _field(block: str, name: str) -> Optional[str]:
    match = re.search(rf"^\s*{re.escape(name)}:\s*(.+?)\s*$", block, re.MULTILINE)
    return match.group(1) if match else None


def _parse_size_bytes(text: Optional[str]) -> Optional[int]:
    """``Size: 16 GB`` -> bytes. Returns None for an empty slot (``No Module Installed``)."""
    if not text:
        return None
    match = re.match(r"^(\d+)\s*(kB|KB|MB|GB|TB)$", text.strip(), re.IGNORECASE)
    if not match:
        return None  # "No Module Installed", "Unknown", ...
    scale = {"kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}[match.group(2).lower()]
    return int(match.group(1)) * scale


def _parse_mtps(text: Optional[str]) -> Optional[float]:
    """``Speed: 6400 MT/s`` -> 6400.0. Older dmidecode reports ``MHz`` for the same quantity."""
    if not text:
        return None
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(MT/s|MHz)$", text.strip(), re.IGNORECASE)
    return float(match.group(1)) if match else None


def _parse_bits(text: Optional[str]) -> Optional[int]:
    """``Data Width: 32 bits`` -> 32."""
    if not text:
        return None
    match = re.match(r"^(\d+)\s*bits?$", text.strip(), re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_dmidecode_memory(text: str) -> DramSpec:
    """Parse the output of ``dmidecode --type 17`` (a "Memory Device" listing).

    Only POPULATED devices are kept: an empty slot reports ``Size: No Module Installed`` and would
    otherwise contribute a phantom channel to the peak. ``Configured Memory Speed`` is preferred over
    ``Speed`` because it is the rate the memory actually RUNS at (the module's rated ``Speed`` can be
    higher than what the controller drives).
    """
    devices: List[MemoryDevice] = []
    for block in re.split(r"\n(?=Memory Device\b)|\n\n(?=Handle )", text):
        if "Memory Device" not in block:
            continue
        size = _parse_size_bytes(_field(block, "Size"))
        if not size:
            continue  # empty slot
        speed = _parse_mtps(_field(block, "Configured Memory Speed")) or _parse_mtps(_field(block, "Speed"))
        width = _parse_bits(_field(block, "Data Width"))
        if speed is None or width is None:
            continue  # unusable entry: no rate or no width to multiply
        devices.append(
            MemoryDevice(size_bytes=size,
                         speed_mtps=speed,
                         data_width_bits=width,
                         memory_type=(_field(block, "Type") or "unknown")))
    return DramSpec(devices=devices)


def read_dmidecode_file(path: str) -> DramSpec:
    """:func:`parse_dmidecode_memory` on a saved dump (``sudo dmidecode -t 17 > dram.txt``)."""
    with open(path, "r") as handle:
        return parse_dmidecode_memory(handle.read())


DMIDECODE_COMMAND = ["dmidecode", "--type", "17"]


def run_dmidecode(sudo: bool = True) -> DramSpec:
    """Run ``dmidecode --type 17`` and parse it. dmidecode reads the DMI table and needs root, so
    ``sudo`` is prepended by default -- this may PROMPT for a password, which is why it is never
    called implicitly. Use :func:`read_dmidecode_file` in an unattended context."""
    import subprocess

    command = (["sudo"] if sudo else []) + DMIDECODE_COMMAND
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return parse_dmidecode_memory(completed.stdout)


def host_dram_spec(dump_path: Optional[str] = None, sudo: bool = True) -> DramSpec:
    """The host DRAM configuration: parsed from ``dump_path`` when given, else read live via
    :func:`run_dmidecode` (which may prompt for a sudo password)."""
    if dump_path is not None:
        return read_dmidecode_file(dump_path)
    return run_dmidecode(sudo=sudo)


def gpu_peak_bytes_per_s(device: int = 0) -> float:
    """Theoretical peak GPU memory bandwidth from the CUDA device properties:
    ``memory_clock x 2 (DDR) x bus_width / 8``. Cross-check: an RTX 4050 Laptop reports 8.001 GHz on
    a 96-bit bus -> 192.0 GB/s, which is NVIDIA's published figure."""
    import cupy  # only on the GPU path

    props = cupy.cuda.runtime.getDeviceProperties(device)
    clock_hz = props["memoryClockRate"] * 1e3  # reported in kHz
    return clock_hz * 2 * props["memoryBusWidth"] / BITS_PER_BYTE
