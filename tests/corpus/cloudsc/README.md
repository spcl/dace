# CloudSC test corpus

The ECMWF `dwarf-p-cloudsc` cloud microphysics kernel, inlined into a single `dace.program`
(`cloudsc.py`). It is callback-free, so `cloudsc_py.to_sdfg()` builds standalone. The result is a
large, wide, deeply nested SDFG (thousands of blocks, nested loop regions many levels deep), which
makes it useful as a scaling test for whole-SDFG analyses and passes.

`generate_data_for_cloudsc.py` provides:

- `build_cloudsc_sdfg(simplify=False)` — the parsed SDFG. The parse takes minutes, so it is memoized
  per process in `PARSED_CLOUDSC` and every caller gets a deepcopy: an SDFG is mutable and every
  consumer transforms it, so handing out the memoized object would leak one test's edits into the
  next. Nothing is written to disk, and under pytest-xdist each worker pays the parse once.
- `generate_cloudsc_inputs(sdfg, seed)` — a physically realistic input set. The physical constants
  and the per-array `[min, max]` ranges are the values from the dwarf's `config-files/input.h5`,
  mirrored here so nothing external is needed. Random inputs would sit on every threshold in the
  kernel, where a harmless floating-point reassociation flips a branch and looks like a bug.
- `run_and_compare(reference, candidate)` — drives both SDFGs on the same inputs and compares every
  output array. With the default IEEE build (`-O0`, no fast-math, no FP contraction) and sequential
  schedules, a value-preserving transformation reproduces the reference bit-for-bit, hence the
  `1e-15` default tolerance.

The grid is small (`klev = klon = 32`) so a compiled run is quick; the input ranges are bounds, not
vertical profiles, so they stay valid at any size.
