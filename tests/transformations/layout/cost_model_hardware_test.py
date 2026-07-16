# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Theoretical peak memory bandwidth, the denominator the LogP memory model is judged against.

dmidecode needs root, so the parser reads a SAVED dump and these tests use synthetic dumps -- they
need neither root nor a GPU. Peak is the sum over populated devices of ``speed x data_width``, which
must be right for both DIMMs and soldered LPDDR (where a per-channel formula gets it wrong)."""
import pytest

from dace.transformation.layout.cost_model.hardware import (parse_dmidecode_memory, read_dmidecode_file)

# Soldered LPDDR5x: 4 packages x 32-bit @ 6400 MT/s -> 128-bit total -> 102.4 GB/s.
LPDDR5X_DUMP = """
Handle 0x0012, DMI type 17, 32 bytes
Memory Device
\tArray Handle: 0x0011
\tSize: 8 GB
\tForm Factor: Row Of Chips
\tLocator: DIMM 0
\tType: LPDDR5
\tSpeed: 6400 MT/s
\tData Width: 32 bits
\tTotal Width: 32 bits
\tConfigured Memory Speed: 6400 MT/s

Handle 0x0013, DMI type 17, 32 bytes
Memory Device
\tArray Handle: 0x0011
\tSize: 8 GB
\tType: LPDDR5
\tSpeed: 6400 MT/s
\tData Width: 32 bits
\tConfigured Memory Speed: 6400 MT/s

Handle 0x0014, DMI type 17, 32 bytes
Memory Device
\tArray Handle: 0x0011
\tSize: 8 GB
\tType: LPDDR5
\tSpeed: 6400 MT/s
\tData Width: 32 bits
\tConfigured Memory Speed: 6400 MT/s

Handle 0x0015, DMI type 17, 32 bytes
Memory Device
\tArray Handle: 0x0011
\tSize: 8 GB
\tType: LPDDR5
\tSpeed: 6400 MT/s
\tData Width: 32 bits
\tConfigured Memory Speed: 6400 MT/s
"""

# Classic desktop: 2 x 64-bit DDR5-5600 DIMMs, plus TWO EMPTY SLOTS that must not be counted.
DDR5_DUMP = """
Handle 0x0020, DMI type 17, 32 bytes
Memory Device
\tSize: 16 GB
\tType: DDR5
\tSpeed: 5600 MT/s
\tData Width: 64 bits
\tTotal Width: 64 bits
\tConfigured Memory Speed: 5600 MT/s

Handle 0x0021, DMI type 17, 32 bytes
Memory Device
\tSize: 16 GB
\tType: DDR5
\tSpeed: 5600 MT/s
\tData Width: 64 bits
\tConfigured Memory Speed: 5600 MT/s

Handle 0x0022, DMI type 17, 32 bytes
Memory Device
\tSize: No Module Installed
\tType: Unknown
\tSpeed: Unknown
\tData Width: Unknown

Handle 0x0023, DMI type 17, 32 bytes
Memory Device
\tSize: No Module Installed
\tType: Unknown
\tSpeed: Unknown
\tData Width: Unknown
"""


def test_soldered_lpddr5x_peak():
    """4 x 32-bit @ 6400 MT/s = 102.4 GB/s. A 'channels x speed x 8 bytes' formula would say 204.8."""
    spec = parse_dmidecode_memory(LPDDR5X_DUMP)
    assert len(spec.devices) == 4
    assert spec.memory_type == "LPDDR5"
    assert spec.total_width_bits == 128
    assert spec.peak_bytes_per_s() == pytest.approx(102.4e9)


def test_ddr5_dimms_peak_ignores_empty_slots():
    """2 x 64-bit @ 5600 MT/s = 89.6 GB/s; the two 'No Module Installed' slots contribute nothing."""
    spec = parse_dmidecode_memory(DDR5_DUMP)
    assert len(spec.devices) == 2  # empty slots dropped
    assert spec.total_width_bits == 128
    assert spec.peak_bytes_per_s() == pytest.approx(89.6e9)


def test_configured_speed_wins_over_rated_speed():
    """The controller may drive a module below its rated speed; the CONFIGURED rate is the real one."""
    dump = """
Memory Device
\tSize: 16 GB
\tType: DDR5
\tSpeed: 5600 MT/s
\tData Width: 64 bits
\tConfigured Memory Speed: 4800 MT/s
"""
    spec = parse_dmidecode_memory(dump)
    assert spec.devices[0].speed_mtps == 4800.0
    assert spec.peak_bytes_per_s() == pytest.approx(4800e6 * 8)


def test_mhz_unit_is_accepted():
    """Older dmidecode reports the same quantity as MHz."""
    dump = """
Memory Device
\tSize: 8 GB
\tType: DDR4
\tSpeed: 3200 MHz
\tData Width: 64 bits
"""
    assert parse_dmidecode_memory(dump).peak_bytes_per_s() == pytest.approx(3200e6 * 8)


def test_unparsable_entries_are_dropped_not_guessed():
    """A device with no usable width/speed must not contribute a phantom peak."""
    dump = """
Memory Device
\tSize: 8 GB
\tType: DDR4
\tSpeed: Unknown
\tData Width: Unknown
"""
    spec = parse_dmidecode_memory(dump)
    assert spec.devices == []
    assert spec.peak_bytes_per_s() == 0.0


def test_read_dmidecode_file(tmp_path):
    path = tmp_path / "dram.txt"
    path.write_text(LPDDR5X_DUMP)
    assert read_dmidecode_file(str(path)).peak_bytes_per_s() == pytest.approx(102.4e9)


if __name__ == "__main__":
    test_soldered_lpddr5x_peak()
    test_ddr5_dimms_peak_ignores_empty_slots()
    test_configured_speed_wins_over_rated_speed()
    test_mhz_unit_is_accepted()
    test_unparsable_entries_are_dropped_not_guessed()
    print("cost_model hardware tests PASS")
