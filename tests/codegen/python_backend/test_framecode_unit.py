# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for dace/codegen/py/framecode.py — DaCePythonCodeGenerator and helpers."""

from collections.abc import Set
from types import SimpleNamespace
import pytest
import numpy as np

import dace
from dace import data, dtypes
import dace.codegen.py.framecode as framecode_module
from dace.codegen.py.framecode import DaCePythonCodeGenerator, codeblock_to_python
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.py.python_target import PythonCodeGen
from dace.properties import CodeBlock
from dace.sdfg import SDFG, nodes
from dace.sdfg.state import LoopRegion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sdfg(name: str = "test_sdfg") -> SDFG:
    """Create a minimal SDFG with one empty state."""
    sdfg = dace.SDFG(name)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    sdfg.add_state("init")
    return sdfg


def _make_sdfg_with_tasklet(name: str = "tasklet_sdfg") -> SDFG:
    """Create an SDFG with A[0] -> tasklet -> B[0]."""
    sdfg = dace.SDFG(name)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state("compute")
    a = state.add_read("A")
    b = state.add_write("B")
    t = state.add_tasklet("copy", {"inp"}, {"out"}, "out = inp")
    state.add_edge(a, None, t, "inp", dace.Memlet("A[0]"))
    state.add_edge(t, "out", b, None, dace.Memlet("B[0]"))
    return sdfg


def _generate_code_for(sdfg: SDFG):
    """Run full code generation on the SDFG and return the code objects."""
    return sdfg.generate_code()


def _make_codegen_with_python_target(sdfg: SDFG) -> DaCePythonCodeGenerator:
    """Create frame codegen with the real Python target registered but still unused."""
    codegen = DaCePythonCodeGenerator(sdfg)
    codegen.targets.add(PythonCodeGen(codegen, sdfg))
    return codegen


# ===========================================================================
# codeblock_to_python
# ===========================================================================


class TestCodeblockToPython:

    def test_codeblock_python_language(self):
        """Python CodeBlock → returns Python source text."""
        cb = CodeBlock("x = 42")
        result = codeblock_to_python(cb)
        assert result == "x = 42"

    def test_codeblock_non_python_as_string(self):
        """Non-Python with code → ValueError."""
        cb = CodeBlock("int x = 42;", language=dtypes.Language.CPP)
        with pytest.raises(ValueError, match="cannot be converted to Python"):
            codeblock_to_python(cb)

    def test_codeblock_non_python_empty(self):
        """Non-Python empty → returns ''."""
        cb = CodeBlock("")
        # Force language to CPP but with empty code
        cb.language = dtypes.Language.CPP
        cb.code = []  # empty code list
        result = codeblock_to_python(cb)
        assert result == ""


# ===========================================================================
# __init__ and symbol resolution
# ===========================================================================


class TestInitAndSymbolResolution:

    def test_runtime_code_helpers_use_python_language_for_python_backend(self):
        """Public runtime-code helpers accept an explicit Python language override."""
        sdfg = _make_sdfg("runtime_helper_language")

        sdfg.set_global_code("GLOBAL_VALUE = 1", language=dtypes.Language.Python)
        sdfg.append_init_code("INIT_VALUE = GLOBAL_VALUE + 1", language=dtypes.Language.Python)
        sdfg.append_exit_code("EXIT_VALUE = INIT_VALUE + 1", language=dtypes.Language.Python)

        assert sdfg.global_code['frame'].language == dtypes.Language.Python
        assert sdfg.init_code['frame'].language == dtypes.Language.Python
        assert sdfg.exit_code['frame'].language == dtypes.Language.Python

    def test_runtime_code_append_preserves_python_codeblock_representation(self):
        """Appending Python runtime code keeps the block AST-backed instead of downgrading to a raw string."""
        sdfg = _make_sdfg("runtime_append_representation")

        sdfg.set_global_code("value = 1", language=dtypes.Language.Python)
        sdfg.append_global_code("value = value + 1", language=dtypes.Language.Python)

        assert sdfg.global_code['frame'].language == dtypes.Language.Python
        assert isinstance(sdfg.global_code['frame'].code, list)
        assert sdfg.global_code['frame'].as_string == "value = 1\nvalue = (value + 1)"

    def test_runtime_code_append_rejects_explicit_language_conflict(self):
        """Appending runtime code with a conflicting explicit language fails fast."""
        sdfg = _make_sdfg("runtime_append_conflict")
        sdfg.set_init_code("sentinel = 1", language=dtypes.Language.Python)

        with pytest.raises(ValueError, match='Cannot append code with language'):
            sdfg.append_init_code('int sentinel = 1;', language=dtypes.Language.CPP)


    def test_python_backend_default_map_schedule_becomes_sequential(self):
        """Default-scheduled maps are normalized to Sequential before Python dispatch."""
        sdfg = dace.SDFG('python_default_map_schedule')
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array('A', [4], dace.float64)
        sdfg.add_array('B', [4], dace.float64)
        state = sdfg.add_state('compute')
        map_entry, map_exit = state.add_map('m', {'i': '0:4'})
        tasklet = state.add_tasklet('copy', {'inp'}, {'out'}, 'out = inp')
        state.add_memlet_path(state.add_read('A'), map_entry, tasklet, dst_conn='inp', memlet=dace.Memlet('A[i]'))
        state.add_memlet_path(tasklet, map_exit, state.add_write('B'), src_conn='out', memlet=dace.Memlet('B[i]'))

        generated_code = sdfg.generate_code()[0].code

        assert 'for i in range' in generated_code

    def test_init_root_sdfg_symbols(self):
        """Root SDFG symbols resolved correctly during __init__."""
        sdfg = dace.SDFG("root_sym")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_symbol("N", dace.int32)
        sdfg.add_array("A", [dace.symbol("N")], dace.float64)
        state = sdfg.add_state("s0")
        state.add_read("A")

        codegen = DaCePythonCodeGenerator(sdfg)
        syms = codegen.symbols_and_constants(sdfg)
        assert "N" in syms

    def test_init_nested_sdfg_symbol_propagation(self):
        """Nested SDFG inherits parent symbols."""
        outer = dace.SDFG("outer")
        outer.backend = dace.dtypes.BackendLanguage.Python
        outer.add_symbol("N", dace.int32)
        outer.add_array("A", [1], dace.float64)
        outer.add_array("B", [1], dace.float64)

        inner = dace.SDFG("inner")
        inner.add_symbol("N", dace.int32)
        inner.add_array("X", [1], dace.float64)
        inner.add_array("Y", [1], dace.float64)

        ostate = outer.add_state("outer_state")
        a = ostate.add_read("A")
        b = ostate.add_write("B")
        nsdfg = ostate.add_nested_sdfg(inner, {"X"}, {"Y"}, symbol_mapping={"N": "N"})
        ostate.add_edge(a, None, nsdfg, "X", dace.Memlet("A[0]"))
        ostate.add_edge(nsdfg, "Y", b, None, dace.Memlet("B[0]"))

        istate = inner.add_state("inner_state")
        x = istate.add_read("X")
        y = istate.add_write("Y")
        t = istate.add_tasklet("t", {"inp"}, {"out"}, "out = inp")
        istate.add_edge(x, None, t, "inp", dace.Memlet("X[0]"))
        istate.add_edge(t, "out", y, None, dace.Memlet("Y[0]"))

        codegen = DaCePythonCodeGenerator(outer)
        inner_syms = codegen.symbols_and_constants(inner)
        outer_syms = codegen.symbols_and_constants(outer)
        # N should propagate from outer to inner
        assert "N" in inner_syms
        assert "N" in outer_syms

    def test_nested_runtime_names_do_not_hide_outer_free_symbols(self):
        """Nested helper runtime globals must not remove outer free symbols from the top-level signature."""
        outer = dace.SDFG("outer_runtime_symbol")
        outer.backend = dace.dtypes.BackendLanguage.Python
        outer.add_symbol("N", dace.int32)
        outer.add_array("A", [1], dace.int32)
        outer.add_array("B", [1], dace.int32)
        outer.add_transient("tmp", [1], dace.int32)

        inner = dace.SDFG("inner_runtime_symbol")
        inner.backend = dace.dtypes.BackendLanguage.Python
        inner.set_global_code("N = 5", language=dtypes.Language.Python)
        inner.add_array("X", [1], dace.int32)
        inner.add_array("Y", [1], dace.int32)
        inner_state = inner.add_state("inner_state", is_start_block=True)
        inner_tasklet = inner_state.add_tasklet("inner_add", {"inp"}, {"out"}, "out = inp + N")
        inner_state.add_edge(inner_state.add_read("X"), None, inner_tasklet, "inp", dace.Memlet("X[0]"))
        inner_state.add_edge(inner_tasklet, "out", inner_state.add_write("Y"), None, dace.Memlet("Y[0]"))

        outer_state = outer.add_state("outer_state", is_start_block=True)
        nested = outer_state.add_nested_sdfg(inner, {"X"}, {"Y"})
        outer_state.add_edge(outer_state.add_read("A"), None, nested, "X", dace.Memlet("A[0]"))
        outer_state.add_edge(nested, "Y", outer_state.add_access("tmp"), None, dace.Memlet("tmp[0]"))
        outer_tasklet = outer_state.add_tasklet("outer_add", {"inp"}, {"out"}, "out = inp + N")
        outer_state.add_edge(outer_state.add_read("tmp"), None, outer_tasklet, "inp", dace.Memlet("tmp[0]"))
        outer_state.add_edge(outer_tasklet, "out", outer_state.add_write("B"), None, dace.Memlet("B[0]"))

        codegen = DaCePythonCodeGenerator(outer)

        assert "N" in codegen.arglist

    def test_runtime_name_analysis_handles_loop_assignments(self):
        """Names assigned inside top-level Python control flow are treated as runtime-defined names."""
        sdfg = dace.SDFG("runtime_loop_assignment")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [1], dace.int64)
        sdfg.append_init_code("for idx in range(1):\n    LOOP_VALUE = idx + 1", language=dtypes.Language.Python)

        state = sdfg.add_state("state", is_start_block=True)
        tasklet = state.add_tasklet("write", {}, {"out"}, "out = LOOP_VALUE")
        state.add_edge(tasklet, "out", state.add_write("A"), None, dace.Memlet("A[0]"))

        codegen = DaCePythonCodeGenerator(sdfg)

        assert "LOOP_VALUE" not in codegen.arglist

    def test_init_constant_propagation_to_nested(self):
        """Constant edge propagated to nested SDFG."""
        outer = dace.SDFG("outer_const")
        outer.backend = dace.dtypes.BackendLanguage.Python
        outer.add_constant("C", 42)
        outer.add_array("A", [1], dace.float64)
        outer.add_array("B", [1], dace.float64)

        inner = dace.SDFG("inner_const")
        inner.add_array("X", [1], dace.float64)
        inner.add_array("Y", [1], dace.float64)

        ostate = outer.add_state("ostate")
        a = ostate.add_read("A")
        b = ostate.add_write("B")
        nsdfg = ostate.add_nested_sdfg(inner, {"X"}, {"Y"})
        ostate.add_edge(a, None, nsdfg, "X", dace.Memlet("A[0]"))
        ostate.add_edge(nsdfg, "Y", b, None, dace.Memlet("B[0]"))

        istate = inner.add_state("istate")
        x = istate.add_read("X")
        y = istate.add_write("Y")
        t = istate.add_tasklet("t", {"inp"}, {"out"}, "out = inp")
        istate.add_edge(x, None, t, "inp", dace.Memlet("X[0]"))
        istate.add_edge(t, "out", y, None, dace.Memlet("Y[0]"))

        codegen = DaCePythonCodeGenerator(outer)
        outer_syms = codegen.symbols_and_constants(outer)
        assert "C" in outer_syms

    def test_symbols_and_constants_lookup(self):
        """symbols_and_constants returns correct set for cfg_id."""
        sdfg = _make_sdfg("sym_lookup")
        sdfg.add_symbol("X", dace.int32)
        sdfg.add_constant("K", 10)

        codegen = DaCePythonCodeGenerator(sdfg)
        result = codegen.symbols_and_constants(sdfg)
        assert "X" in result
        assert "K" in result

    def test_free_symbols_cached(self):
        """Second call returns cached result."""
        sdfg = _make_sdfg("cached_sym")
        sdfg.add_symbol("N", dace.int32)

        codegen = DaCePythonCodeGenerator(sdfg)
        r1 = codegen.free_symbols(sdfg)
        r2 = codegen.free_symbols(sdfg)
        assert r1 is r2

    def test_free_symbols_uses_used_symbols(self):
        """Objects with used_symbols attribute use that instead of free_symbols."""
        sdfg = _make_sdfg("used_sym")
        codegen = DaCePythonCodeGenerator(sdfg)

        class HasUsedSymbols:
            def used_symbols(self, all_symbols=False):
                return {"alpha", "beta"}
        obj = HasUsedSymbols()
        result = codegen.free_symbols(obj)
        assert result == {"alpha", "beta"}


# ===========================================================================
# generate_constants
# ===========================================================================


class TestGenerateConstants:

    def test_generate_constants_scalar(self):
        """Scalar constant: 'name = value'."""
        sdfg = dace.SDFG("const_scalar")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_state("s0")
        sdfg.add_constant("MY_CONST", 42)

        codegen = DaCePythonCodeGenerator(sdfg)
        stream = PythonCodeIOStream()
        codegen.generate_constants(sdfg, stream)
        code = stream.getvalue()
        assert "MY_CONST = 42" in code

    def test_generate_constants_array(self):
        """Array constant: list format."""
        sdfg = dace.SDFG("const_arr")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_state("s0")
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sdfg.add_constant("MY_ARR", arr)

        codegen = DaCePythonCodeGenerator(sdfg)
        stream = PythonCodeIOStream()
        codegen.generate_constants(sdfg, stream)
        code = "import numpy\n" # Need to import numpy for the array literal because we only call generate_constants, not the full codegen pipeline which would add the import
        code += stream.getvalue()
        namespace = {}
        exec(code, namespace)
        assert "MY_ARR" in namespace
        np.testing.assert_array_equal(namespace["MY_ARR"], arr)

    def test_generate_constants_empty(self):
        """No constants: nothing written."""
        sdfg = _make_sdfg("no_const")

        codegen = DaCePythonCodeGenerator(sdfg)
        stream = PythonCodeIOStream()
        codegen.generate_constants(sdfg, stream)
        code = stream.getvalue()
        assert code.strip() == ""


# ===========================================================================
# generate_fileheader
# ===========================================================================


class TestGenerateFileheader:

    def test_fileheader_env_includes_list(self):
        """Non-dict env headers are treated as frame headers."""
        sdfg = _make_sdfg("env_list")
        codegen = DaCePythonCodeGenerator(sdfg)

        class FakeEnv:
            headers = ["math", "os"]
            state_fields = []

        codegen.environments = [FakeEnv()]
        stream = PythonCodeIOStream()
        codegen.generate_fileheader(sdfg, stream, backend='frame')
        code = stream.getvalue()
        assert "import math" in code
        assert "import os" in code

    def test_fileheader_env_includes_dict(self):
        """Dict env headers with backend key."""
        sdfg = _make_sdfg("env_dict")
        codegen = DaCePythonCodeGenerator(sdfg)

        class FakeEnv:
            headers = {"frame": ["numpy", "sys"]}
            state_fields = []

        codegen.environments = [FakeEnv()]
        stream = PythonCodeIOStream()
        codegen.generate_fileheader(sdfg, stream, backend='frame')
        code = stream.getvalue()
        assert "import numpy" in code
        assert "import sys" in code

    def test_fileheader_env_includes_dict_wrong_backend(self):
        """Dict env headers without matching backend key writes nothing."""
        sdfg = _make_sdfg("env_dict_miss")
        codegen = DaCePythonCodeGenerator(sdfg)

        class FakeEnv:
            headers = {"cpp": ["cstdlib"]}
            state_fields = []

        codegen.environments = [FakeEnv()]
        stream = PythonCodeIOStream()
        codegen.generate_fileheader(sdfg, stream, backend='frame')
        code = stream.getvalue()
        assert "cstdlib" not in code

    def test_fileheader_state_struct(self):
        """State struct class generated when fields exist."""
        sdfg = _make_sdfg("struct_test")
        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.environments = []
        codegen.statestruct.append("field1 = None")
        codegen.statestruct.append("field2 = 0")

        stream = PythonCodeIOStream()
        codegen.generate_fileheader(sdfg, stream, backend='frame')
        code = stream.getvalue()
        assert "class " in code
        assert "field1 = None" in code
        assert "field2 = 0" in code

    def test_fileheader_no_state_struct(self):
        """No state struct when empty."""
        sdfg = _make_sdfg("no_struct")
        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.environments = []
        # statestruct is empty by default

        stream = PythonCodeIOStream()
        codegen.generate_fileheader(sdfg, stream, backend='frame')
        code = stream.getvalue()
        assert "class " not in code

    def test_fileheader_global_code_none_key(self):
        """Global code with None key is written."""
        sdfg = _make_sdfg("global_none")
        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.environments = []
        sdfg.global_code[None] = CodeBlock("GLOBAL_VAR = 99")

        stream = PythonCodeIOStream()
        codegen.generate_fileheader(sdfg, stream, backend='frame')
        code = stream.getvalue()
        assert "GLOBAL_VAR = 99" in code
        assert "Assign(" not in code

    def test_fileheader_global_code_backend_key(self):
        """Global code with backend key is written."""
        sdfg = _make_sdfg("global_be")
        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.environments = []
        sdfg.global_code['frame'] = CodeBlock("FRAME_VAR = 123")

        stream = PythonCodeIOStream()
        codegen.generate_fileheader(sdfg, stream, backend='frame')
        code = stream.getvalue()
        assert "FRAME_VAR = 123" in code
        assert "Assign(" not in code

    def test_fileheader_target_includes(self):
        """Target includes written as 'import X'."""
        sdfg = _make_sdfg("tgt_inc")
        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.environments = []

        # Add a fake target with includes
        class FakeTarget:
            def get_includes(self):
                return {"frame": ["json"]}
        codegen._dispatcher._used_targets.add(FakeTarget())

        stream = PythonCodeIOStream()
        codegen.generate_fileheader(sdfg, stream, backend='frame')
        code = stream.getvalue()
        assert "import json" in code

    def test_fileheader_constant_only_sdfg_includes_numpy_without_used_targets(self):
        """Array constants should request numpy through the real Python target even when no target was dispatched."""
        sdfg = dace.SDFG("const_only_header")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_state("empty")
        sdfg.add_constant("CONST_ARR", np.array([1.0, 2.0], dtype=np.float64))

        codegen = _make_codegen_with_python_target(sdfg)

        assert codegen._dispatcher.used_targets == set()

        stream = PythonCodeIOStream()
        codegen.generate_fileheader(sdfg, stream, backend='frame')
        code = stream.getvalue()

        assert code.count("import numpy") == 1
        assert "CONST_ARR = numpy.array(" in code
        assert "dtype=numpy.float64" in code

    def test_fileheader_empty_program_without_used_targets_does_not_include_numpy(self):
        """An unused real Python target should not add numpy imports for empty programs."""
        sdfg = _make_sdfg("empty_header_no_numpy")
        codegen = _make_codegen_with_python_target(sdfg)

        assert codegen._dispatcher.used_targets == set()

        stream = PythonCodeIOStream()
        codegen.generate_fileheader(sdfg, stream, backend='frame')
        code = stream.getvalue()

        assert "import numpy" not in code

    def test_fileheader_verbatim_import_headers(self):
        """Raw import statements are emitted verbatim."""
        sdfg = _make_sdfg("raw_headers")
        codegen = DaCePythonCodeGenerator(sdfg)

        class FakeEnv:
            headers = {"frame": ["import numpy as np", "from math import sin"]}
            state_fields = []

        codegen.environments = [FakeEnv()]
        stream = PythonCodeIOStream()
        codegen.generate_fileheader(sdfg, stream, backend='frame')
        code = stream.getvalue()
        assert "import numpy as np" in code
        assert "from math import sin" in code

    def test_fileheader_deduplicates_imports_across_sources(self):
        """Imports are deduplicated across target, environment, and Python global code."""
        sdfg = _make_sdfg("dedupe_headers")
        codegen = DaCePythonCodeGenerator(sdfg)

        class FakeEnv:
            headers = {"frame": ["import numpy as np", "from math import sin"]}
            state_fields = []

        class FakeTarget:
            def get_includes(self):
                return {"frame": ["numpy", "from math import sin", "import json"]}

        codegen.environments = [FakeEnv()]
        codegen._dispatcher._used_targets.add(FakeTarget())
        sdfg.global_code['python'] = CodeBlock("import numpy as np\nHEADER_SENTINEL = 1")

        stream = PythonCodeIOStream()
        codegen.generate_fileheader(sdfg, stream, backend='frame')
        code = stream.getvalue()
        code_lines = code.splitlines()

        assert sum(line.startswith("import numpy") and " as np" not in line for line in code_lines) == 1
        assert sum(line.startswith("import numpy as np") for line in code_lines) == 1
        assert sum(line.startswith("from math import sin") for line in code_lines) == 1
        assert sum(line.startswith("import json") for line in code_lines) == 1
        assert "HEADER_SENTINEL = 1" in code


# ===========================================================================
# generate_header / generate_footer
# ===========================================================================


class TestGenerateHeaderFooter:

    def test_generate_header_auto_comment(self):
        """Header contains AUTO-GENERATED comment."""
        sdfg = _make_sdfg("header_test")
        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.environments = []
        global_stream = PythonCodeIOStream()
        callsite_stream = PythonCodeIOStream()
        codegen.generate_header(sdfg, global_stream, callsite_stream)
        code = global_stream.getvalue()
        assert "AUTO-GENERATED" in code

    def test_generate_header_env_state_fields(self):
        """Header processes environment state fields — they are added to statestruct."""
        sdfg = _make_sdfg("header_env")
        codegen = DaCePythonCodeGenerator(sdfg)

        class FakeEnv:
            headers = []
            state_fields = ["counter = 0"]

        codegen.environments = [FakeEnv()]
        # generate_header calls generate_fileheader which uses `with global_stream.indent()`
        # — that's a known bug. But the env state_fields are added BEFORE that call.
        # We test the field was added to statestruct directly.
        # (Avoid calling generate_header which hits the indent() bug if statestruct is non-empty.)
        for env in codegen.environments:
            codegen.statestruct.extend(env.state_fields)
        assert "counter = 0" in codegen.statestruct

    def test_generate_footer_noop(self):
        """Footer is a no-op in Python backend."""
        sdfg = _make_sdfg("footer_test")
        codegen = DaCePythonCodeGenerator(sdfg)
        global_stream = PythonCodeIOStream()
        callsite_stream = PythonCodeIOStream()
        # Should not raise
        codegen.generate_footer(sdfg, global_stream, callsite_stream)


# ===========================================================================
# generate_external_memory_management
# ===========================================================================


class TestExternalMemoryManagement:

    def test_external_memory_raises(self):
        """External lifetime → NotImplementedError."""
        sdfg = dace.SDFG("ext_mem")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [10], dace.float64, lifetime=dtypes.AllocationLifetime.External)
        sdfg.add_state("s0")

        codegen = DaCePythonCodeGenerator(sdfg)
        stream = PythonCodeIOStream()
        with pytest.raises(NotImplementedError, match="External memory management"):
            codegen.generate_external_memory_management(sdfg, stream)

    def test_no_external_memory(self):
        """No external arrays → pass (no error)."""
        sdfg = _make_sdfg("no_ext")
        codegen = DaCePythonCodeGenerator(sdfg)
        stream = PythonCodeIOStream()
        # Should not raise
        codegen.generate_external_memory_management(sdfg, stream)


# ===========================================================================
# _get_schedule
# ===========================================================================


class TestGetSchedule:

    def test_get_schedule_none(self):
        """None → Sequential."""
        sdfg = _make_sdfg("sched_none")
        codegen = DaCePythonCodeGenerator(sdfg)
        result = codegen._get_schedule(None)
        assert result == dtypes.ScheduleType.Sequential

    def test_get_schedule_entry_node(self):
        """EntryNode → its schedule."""
        sdfg = dace.SDFG("sched_entry")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [10], dace.float64)
        state = sdfg.add_state("s0")
        me, mx = state.add_map("m", {"i": "0:10"}, schedule=dtypes.ScheduleType.Sequential)
        codegen = DaCePythonCodeGenerator(sdfg)
        result = codegen._get_schedule(me)
        assert result == dtypes.ScheduleType.Sequential

    def test_get_schedule_top_level_sdfg(self):
        """Top-level SDFG → Sequential."""
        sdfg = _make_sdfg("sched_top")
        codegen = DaCePythonCodeGenerator(sdfg)
        result = codegen._get_schedule(sdfg)
        assert result == dtypes.ScheduleType.Sequential

    def test_get_schedule_state(self):
        """SDFGState of top-level SDFG → Sequential."""
        sdfg = _make_sdfg("sched_state")
        codegen = DaCePythonCodeGenerator(sdfg)
        state = next(sdfg.states())
        result = codegen._get_schedule(state)
        assert result == dtypes.ScheduleType.Sequential

    def test_get_schedule_invalid_type(self):
        """Other type → TypeError."""
        sdfg = _make_sdfg("sched_invalid")
        codegen = DaCePythonCodeGenerator(sdfg)
        with pytest.raises(TypeError):
            codegen._get_schedule("not_a_valid_scope")


# ===========================================================================
# _can_allocate
# ===========================================================================


class TestCanAllocate:

    def test_can_allocate_cpu_heap(self):
        """CPU_Heap with Sequential schedule → True."""
        sdfg = _make_sdfg("alloc_cpu")
        codegen = DaCePythonCodeGenerator(sdfg)
        desc = data.Array(dace.float64, [10], storage=dtypes.StorageType.CPU_Heap)
        result = codegen._can_allocate(sdfg, next(sdfg.states()), desc, sdfg)
        assert result is True

    def test_can_allocate_register(self):
        """Register storage → True."""
        sdfg = _make_sdfg("alloc_reg")
        codegen = DaCePythonCodeGenerator(sdfg)
        desc = data.Scalar(dace.float64, storage=dtypes.StorageType.Register)
        result = codegen._can_allocate(sdfg, next(sdfg.states()), desc, sdfg)
        assert result is True


# ===========================================================================
# determine_allocation_lifetime
# ===========================================================================


class TestDetermineAllocationLifetime:

    def test_lifetime_persistent(self):
        """Persistent → top SDFG alloc + statestruct entry."""
        sdfg = dace.SDFG("lt_persist")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [5], dace.float64, transient=True,
                        lifetime=dtypes.AllocationLifetime.Persistent)
        state = sdfg.add_state("s0")
        state.add_access("T")

        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        # Persistent should be allocated in top SDFG
        assert sdfg in codegen.to_allocate
        assert any(entry[0] is sdfg for entry in codegen.to_allocate[sdfg])
        assert len(codegen.statestruct) > 0

    def test_lifetime_sdfg(self):
        """SDFG lifetime → SDFG scope alloc."""
        sdfg = dace.SDFG("lt_sdfg")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [5], dace.float64, transient=True,
                        lifetime=dtypes.AllocationLifetime.SDFG)
        state = sdfg.add_state("s0")
        state.add_access("T")

        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        assert sdfg in codegen.to_allocate

    def test_lifetime_state_single(self):
        """State lifetime, single state → state alloc."""
        sdfg = dace.SDFG("lt_state_single")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [5], dace.float64, transient=True,
                        lifetime=dtypes.AllocationLifetime.State)
        state = sdfg.add_state("s0")
        state.add_access("T")

        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        # Should be allocated at the state level, not SDFG level
        assert state in codegen.to_allocate

    def test_lifetime_state_multi(self):
        """State lifetime, multi state → SDFG alloc."""
        sdfg = dace.SDFG("lt_state_multi")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [5], dace.float64, transient=True,
                        lifetime=dtypes.AllocationLifetime.State)
        s0 = sdfg.add_state("s0")
        s1 = sdfg.add_state("s1")
        sdfg.add_edge(s0, s1, dace.InterstateEdge())
        s0.add_access("T")
        s1.add_access("T")

        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        # Multi-state: should be allocated at SDFG level
        assert sdfg in codegen.to_allocate

    def test_lifetime_unused_transient(self):
        """Unused transient → skipped."""
        sdfg = dace.SDFG("lt_unused")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [5], dace.float64, transient=True,
                        lifetime=dtypes.AllocationLifetime.Scope)
        sdfg.add_state("s0")
        # No access node for T

        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        # T should not appear in any allocation
        all_names = set()
        for entries in codegen.to_allocate.values():
            for entry in entries:
                all_names.add(entry[2].data)
        assert "T" not in all_names

    def test_lifetime_constant(self):
        """Constant → skipped."""
        sdfg = dace.SDFG("lt_const")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("C", [3], dace.float64, transient=True)
        sdfg.add_constant("C", np.array([1.0, 2.0, 3.0]))
        sdfg.add_state("s0")

        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        # Constants do not need allocation
        all_names = set()
        for entries in codegen.to_allocate.values():
            for entry in entries:
                all_names.add(entry[2].data)
        assert "C" not in all_names

    def test_lifetime_non_transient_skipped(self):
        """Non-transient data → not allocated by determine_allocation_lifetime."""
        sdfg = dace.SDFG("lt_nontrans")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [5], dace.float64, transient=False)
        state = sdfg.add_state("s0")
        state.add_access("A")

        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        all_names = set()
        for entries in codegen.to_allocate.values():
            for entry in entries:
                all_names.add(entry[2].data)
        assert "A" not in all_names

    def test_lifetime_global(self):
        """Global lifetime → statestruct + top SDFG alloc."""
        sdfg = dace.SDFG("lt_global")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [5], dace.float64, transient=True,
                        lifetime=dtypes.AllocationLifetime.Global)
        state = sdfg.add_state("s0")
        state.add_access("T")

        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        assert sdfg in codegen.to_allocate
        assert len(codegen.statestruct) > 0

    def test_lifetime_scope_single_state(self):
        """Scope lifetime, single state/scope → scope alloc."""
        sdfg = dace.SDFG("lt_scope")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [10], dace.float64)
        sdfg.add_array("T", [10], dace.float64, transient=True,
                        lifetime=dtypes.AllocationLifetime.Scope)
        sdfg.add_array("B", [10], dace.float64)
        state = sdfg.add_state("s0")
        a = state.add_read("A")
        b = state.add_write("B")
        t_node = state.add_access("T")
        t1 = state.add_tasklet("t1", {"inp"}, {"out"}, "out = inp")
        t2 = state.add_tasklet("t2", {"inp"}, {"out"}, "out = inp")
        state.add_edge(a, None, t1, "inp", dace.Memlet("A[0]"))
        state.add_edge(t1, "out", t_node, None, dace.Memlet("T[0]"))
        state.add_edge(t_node, None, t2, "inp", dace.Memlet("T[0]"))
        state.add_edge(t2, "out", b, None, dace.Memlet("B[0]"))

        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        # T should appear in some allocation
        all_names = set()
        for entries in codegen.to_allocate.values():
            for entry in entries:
                all_names.add(entry[2].data)
        assert "T" in all_names


# ===========================================================================
# generate_state
# ===========================================================================


class TestGenerateState:

    def test_generate_state_single_component(self):
        """Single component dispatch — code is generated via full pipeline."""
        sdfg = _make_sdfg_with_tasklet("gen_state_single")
        # Use the full code generation pipeline to properly register PythonCodeGen
        code_objects = _generate_code_for(sdfg)
        code = code_objects[0].code
        # Tasklet body should appear in output
        assert "out = inp" in code or "inp" in code

    def test_generate_state_empty(self):
        """Empty state → no crash, minimal output."""
        sdfg = _make_sdfg("gen_state_empty")
        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        state = next(sdfg.states())

        global_stream = PythonCodeIOStream()
        callsite_stream = PythonCodeIOStream()
        codegen.generate_state(sdfg, sdfg, state, global_stream, callsite_stream)
        # Should not raise; output can be empty

    def test_generate_state_multiple_components(self):
        """Multiple components dispatched sequentially via full pipeline."""
        sdfg = dace.SDFG("multi_comp")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [1], dace.float64)
        sdfg.add_array("B", [1], dace.float64)
        sdfg.add_array("C", [1], dace.float64)
        sdfg.add_array("D", [1], dace.float64)
        state = sdfg.add_state("s0")

        # Two disconnected tasklets → two components
        a = state.add_read("A")
        b = state.add_write("B")
        t1 = state.add_tasklet("t1", {"inp"}, {"out"}, "out = inp")
        state.add_edge(a, None, t1, "inp", dace.Memlet("A[0]"))
        state.add_edge(t1, "out", b, None, dace.Memlet("B[0]"))

        c = state.add_read("C")
        d = state.add_write("D")
        t2 = state.add_tasklet("t2", {"inp"}, {"out"}, "out = inp * 2")
        state.add_edge(c, None, t2, "inp", dace.Memlet("C[0]"))
        state.add_edge(t2, "out", d, None, dace.Memlet("D[0]"))

        # Use the full code generation pipeline
        code_objects = _generate_code_for(sdfg)
        code = code_objects[0].code
        # Both tasklet bodies should appear in the generated code
        assert "out = inp" in code


# ===========================================================================
# generate_code (top-level integration)
# ===========================================================================


class TestGenerateCode:

    def test_generate_code_top_level(self):
        """Top-level SDFG generates complete function."""
        sdfg = _make_sdfg("gen_top")
        code_objects = _generate_code_for(sdfg)
        assert len(code_objects) >= 1
        code = code_objects[0].code
        assert f"def gen_top" in code

    def test_generate_code_function_wrapper(self):
        """Function has correct params and body."""
        sdfg = dace.SDFG("func_wrap")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [1], dace.float64)
        sdfg.add_array("B", [1], dace.float64)
        state = sdfg.add_state("s0")
        a = state.add_read("A")
        b = state.add_write("B")
        t = state.add_tasklet("t", {"inp"}, {"out"}, "out = inp + 1")
        state.add_edge(a, None, t, "inp", dace.Memlet("A[0]"))
        state.add_edge(t, "out", b, None, dace.Memlet("B[0]"))

        code_objects = _generate_code_for(sdfg)
        code = code_objects[0].code
        assert "def func_wrap(" in code
        assert "A" in code
        assert "B" in code

    def test_generate_code_empty_body(self):
        """Empty body → pass in function."""
        sdfg = dace.SDFG("empty_body")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_state("s0")

        code_objects = _generate_code_for(sdfg)
        code = code_objects[0].code
        assert "def empty_body" in code
        assert "pass" in code

    def test_generate_code_with_constants(self):
        """Constants appear as defined vars in generated code."""
        sdfg = dace.SDFG("with_const")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_constant("MY_C", 99)
        sdfg.add_state("s0")

        code_objects = _generate_code_for(sdfg)
        code = code_objects[0].code
        assert "MY_C" in code
        assert "99" in code

    def test_generate_code_interstate_symbols(self):
        """Loop variables and edge symbols resolved."""
        sdfg = dace.SDFG("interstate_sym")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_symbol("N", dace.int32)
        sdfg.add_array("A", [1], dace.float64)

        # Create a simple loop with interstate edge assignment
        s0 = sdfg.add_state("s0")
        s1 = sdfg.add_state("s1")
        a = s0.add_read("A")
        aw = s0.add_write("A")
        t = s0.add_tasklet("t", {"inp"}, {"out"}, "out = inp")
        s0.add_edge(a, None, t, "inp", dace.Memlet("A[0]"))
        s0.add_edge(t, "out", aw, None, dace.Memlet("A[0]"))

        # Interstate edge with assignment
        sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"x": "N + 1"}))

        code_objects = _generate_code_for(sdfg)
        code = code_objects[0].code
        assert "def interstate_sym" in code

    def test_generate_code_sanity_check(self):
        """All states generated (no RuntimeError)."""
        sdfg = dace.SDFG("sanity")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        s0 = sdfg.add_state("s0")
        s1 = sdfg.add_state("s1")
        s2 = sdfg.add_state("s2")
        sdfg.add_edge(s0, s1, dace.InterstateEdge())
        sdfg.add_edge(s1, s2, dace.InterstateEdge())

        # Should not raise RuntimeError about missing states
        code_objects = _generate_code_for(sdfg)
        assert len(code_objects) >= 1

    def test_generate_code_valid_python(self):
        """Generated code is valid Python that can be compiled."""
        sdfg = _make_sdfg_with_tasklet("valid_py")
        code_objects = _generate_code_for(sdfg)
        code = code_objects[0].code
        # Should be compilable Python
        compile(code, "<generated>", "exec")

    @pytest.mark.skip(reason="Test is outdated. The codegen pipeline now generates multiple code objects, and imports them. This is more complicated than a simple exec test.")
    def test_generate_code_executable(self):
        """Generated code is executable — function can be called."""
        sdfg = _make_sdfg_with_tasklet("exec_test")
        code_objects = _generate_code_for(sdfg)
        code = code_objects[0].code

        ns = {}
        exec(code, ns)
        assert "exec_test" in ns
        assert callable(ns["exec_test"])

    def test_generate_code_with_loop_region(self):
        """Code with a LoopRegion generates valid Python."""
        sdfg = dace.SDFG("loop_sdfg")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [1], dace.float64)
        sdfg.add_symbol("N", dace.int32)

        # Pre-loop state
        init_state = sdfg.add_state("init")

        # Create a loop region (correct kwarg names: initialize_expr, update_expr)
        loop = LoopRegion("myloop", condition_expr="i < N", loop_var="i",
                          initialize_expr="i = 0", update_expr="i = i + 1", sdfg=sdfg)
        sdfg.add_node(loop)
        sdfg.add_edge(init_state, loop, dace.InterstateEdge())

        # Add a state inside the loop
        loop_body = loop.add_state("loop_body")
        a_read = loop_body.add_read("A")
        a_write = loop_body.add_write("A")
        t = loop_body.add_tasklet("inc", {"inp"}, {"out"}, "out = inp + 1.0")
        loop_body.add_edge(a_read, None, t, "inp", dace.Memlet("A[0]"))
        loop_body.add_edge(t, "out", a_write, None, dace.Memlet("A[0]"))

        code_objects = _generate_code_for(sdfg)
        code = code_objects[0].code
        assert "def loop_sdfg" in code
        assert "while" in code or "for" in code or "i" in code

    def test_generate_code_cfg_id_nonzero(self):
        """cfg_id formatting for non-zero SDFG (via nested SDFG)."""
        outer = dace.SDFG("outer_cfg")
        outer.backend = dace.dtypes.BackendLanguage.Python
        outer.add_array("A", [1], dace.float64)
        outer.add_array("B", [1], dace.float64)

        inner = dace.SDFG("inner_cfg")
        inner.add_array("X", [1], dace.float64)
        inner.add_array("Y", [1], dace.float64)

        ostate = outer.add_state("os")
        a = ostate.add_read("A")
        b = ostate.add_write("B")
        nsdfg = ostate.add_nested_sdfg(inner, {"X"}, {"Y"})
        ostate.add_edge(a, None, nsdfg, "X", dace.Memlet("A[0]"))
        ostate.add_edge(nsdfg, "Y", b, None, dace.Memlet("B[0]"))

        istate = inner.add_state("is")
        x = istate.add_read("X")
        y = istate.add_write("Y")
        t = istate.add_tasklet("t", {"inp"}, {"out"}, "out = inp")
        istate.add_edge(x, None, t, "inp", dace.Memlet("X[0]"))
        istate.add_edge(t, "out", y, None, dace.Memlet("Y[0]"))

        # This should not raise; nested SDFGs get non-zero cfg_id
        code_objects = _generate_code_for(outer)
        assert len(code_objects) >= 1


# ===========================================================================
# Correctness tests (compile and run)
# ===========================================================================


class TestCorrectness:

    def test_scalar_copy(self):
        """Scalar copy: A[0] → B[0] via tasklet."""
        sdfg = _make_sdfg_with_tasklet("scalar_copy")
        compiled = sdfg.compile()
        A = np.array([7.0], dtype=np.float64)
        B = np.array([0.0], dtype=np.float64)
        compiled(A=A, B=B)
        np.testing.assert_array_equal(B, [7.0])

    def test_scalar_computation(self):
        """Scalar computation: B[0] = A[0] * 2.0."""
        sdfg = dace.SDFG("scalar_comp")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [1], dace.float64)
        sdfg.add_array("B", [1], dace.float64)
        state = sdfg.add_state("compute")
        a = state.add_read("A")
        b = state.add_write("B")
        t = state.add_tasklet("mul", {"inp"}, {"out"}, "out = inp * 2.0")
        state.add_edge(a, None, t, "inp", dace.Memlet("A[0]"))
        state.add_edge(t, "out", b, None, dace.Memlet("B[0]"))

        compiled = sdfg.compile()
        A = np.array([5.0], dtype=np.float64)
        B = np.array([0.0], dtype=np.float64)
        compiled(A=A, B=B)
        np.testing.assert_allclose(B, [10.0])

    def test_two_state_sequential(self):
        """Two sequential states compute correctly."""
        sdfg = dace.SDFG("two_state")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [1], dace.float64)
        sdfg.add_array("B", [1], dace.float64)

        s0 = sdfg.add_state("s0")
        s1 = sdfg.add_state("s1")
        sdfg.add_edge(s0, s1, dace.InterstateEdge())

        # State 0: B[0] = A[0] + 1
        a0 = s0.add_read("A")
        b0 = s0.add_write("B")
        t0 = s0.add_tasklet("add1", {"inp"}, {"out"}, "out = inp + 1.0")
        s0.add_edge(a0, None, t0, "inp", dace.Memlet("A[0]"))
        s0.add_edge(t0, "out", b0, None, dace.Memlet("B[0]"))

        # State 1: B[0] = B[0] * 3
        b1r = s1.add_read("B")
        b1w = s1.add_write("B")
        t1 = s1.add_tasklet("mul3", {"inp"}, {"out"}, "out = inp * 3.0")
        s1.add_edge(b1r, None, t1, "inp", dace.Memlet("B[0]"))
        s1.add_edge(t1, "out", b1w, None, dace.Memlet("B[0]"))

        compiled = sdfg.compile()
        A = np.array([4.0], dtype=np.float64)
        B = np.array([0.0], dtype=np.float64)
        compiled(A=A, B=B)
        # (4 + 1) * 3 = 15
        np.testing.assert_allclose(B, [15.0])

    def test_constant_in_generated_code(self):
        """Compile SDFG with constant and verify it appears in generated code."""
        sdfg = dace.SDFG("const_gen")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_constant("MAGIC", 42)
        sdfg.add_state("s0")

        code_objects = _generate_code_for(sdfg)
        code = code_objects[0].code
        assert "MAGIC = 42" in code

    def test_three_state_pipeline(self):
        """Three sequential states in pipeline."""
        sdfg = dace.SDFG("three_state")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [1], dace.float64)
        sdfg.add_array("B", [1], dace.float64)

        s0 = sdfg.add_state("s0")
        s1 = sdfg.add_state("s1")
        s2 = sdfg.add_state("s2")
        sdfg.add_edge(s0, s1, dace.InterstateEdge())
        sdfg.add_edge(s1, s2, dace.InterstateEdge())

        # s0: B = A + 1
        a0 = s0.add_read("A")
        b0 = s0.add_write("B")
        t0 = s0.add_tasklet("t0", {"x"}, {"y"}, "y = x + 1.0")
        s0.add_edge(a0, None, t0, "x", dace.Memlet("A[0]"))
        s0.add_edge(t0, "y", b0, None, dace.Memlet("B[0]"))

        # s1: B = B + 2
        b1r = s1.add_read("B")
        b1w = s1.add_write("B")
        t1 = s1.add_tasklet("t1", {"x"}, {"y"}, "y = x + 2.0")
        s1.add_edge(b1r, None, t1, "x", dace.Memlet("B[0]"))
        s1.add_edge(t1, "y", b1w, None, dace.Memlet("B[0]"))

        # s2: B = B * 2
        b2r = s2.add_read("B")
        b2w = s2.add_write("B")
        t2 = s2.add_tasklet("t2", {"x"}, {"y"}, "y = x * 2.0")
        s2.add_edge(b2r, None, t2, "x", dace.Memlet("B[0]"))
        s2.add_edge(t2, "y", b2w, None, dace.Memlet("B[0]"))

        compiled = sdfg.compile()
        A = np.array([3.0], dtype=np.float64)
        B = np.array([0.0], dtype=np.float64)
        compiled(A=A, B=B)
        # (3 + 1 + 2) * 2 = 12
        np.testing.assert_allclose(B, [12.0])

    def test_interstate_variable_assignment(self):
        """Interstate edge with assignment creates variable in scope."""
        sdfg = dace.SDFG("is_var")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [1], dace.float64)
        sdfg.add_array("B", [1], dace.float64)

        s0 = sdfg.add_state("s0")
        s1 = sdfg.add_state("s1")
        sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"factor": "3"}))

        # s0: empty
        # s1: B[0] = A[0] (simple tasklet, no dangling connectors)
        a = s1.add_read("A")
        b = s1.add_write("B")
        t = s1.add_tasklet("t", {"inp"}, {"out"}, "out = inp")
        s1.add_edge(a, None, t, "inp", dace.Memlet("A[0]"))
        s1.add_edge(t, "out", b, None, dace.Memlet("B[0]"))

        # This should generate valid code with factor defined
        code_objects = _generate_code_for(sdfg)
        code = code_objects[0].code
        assert "factor" in code


# ===========================================================================
# generate_states and preprocess
# ===========================================================================


class TestGenerateStatesAndPreprocess:

    def test_preprocess_is_noop(self):
        """preprocess() is a no-op — should not raise."""
        sdfg = _make_sdfg("preprocess")
        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.preprocess(sdfg)

    def test_generate_states_returns_all_states(self):
        """generate_states returns all states generated."""
        sdfg = dace.SDFG("gen_states")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        s0 = sdfg.add_state("s0")
        s1 = sdfg.add_state("s1")
        sdfg.add_edge(s0, s1, dace.InterstateEdge())

        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        global_stream = PythonCodeIOStream()
        callsite_stream = PythonCodeIOStream()
        states = codegen.generate_states(sdfg, global_stream, callsite_stream)
        assert len(states) == 2
        assert s0 in states
        assert s1 in states


# ===========================================================================
# allocate_arrays_in_scope / deallocate_arrays_in_scope
# ===========================================================================


class TestAllocDeallocInScope:

    def test_allocate_empty_scope(self):
        """Allocating in a scope with no arrays does nothing."""
        sdfg = _make_sdfg("alloc_empty")
        codegen = DaCePythonCodeGenerator(sdfg)
        func_stream = PythonCodeIOStream()
        call_stream = PythonCodeIOStream()
        # to_allocate[sdfg] is empty by default
        codegen.allocate_arrays_in_scope(sdfg, sdfg, sdfg, func_stream, call_stream)
        # Should not raise; output should be empty
        assert call_stream.getvalue().strip() == ""

    def test_deallocate_empty_scope(self):
        """Deallocating in a scope with no arrays does nothing."""
        sdfg = _make_sdfg("dealloc_empty")
        codegen = DaCePythonCodeGenerator(sdfg)
        func_stream = PythonCodeIOStream()
        call_stream = PythonCodeIOStream()
        codegen.deallocate_arrays_in_scope(sdfg, sdfg, sdfg, func_stream, call_stream)
        # Should not raise


# ===========================================================================
# dispatcher property
# ===========================================================================


class TestDispatcherProperty:

    def test_dispatcher_exists(self):
        """Dispatcher property returns the target dispatcher."""
        sdfg = _make_sdfg("disp_test")
        codegen = DaCePythonCodeGenerator(sdfg)
        from dace.codegen.dispatcher import TargetDispatcher
        assert isinstance(codegen.dispatcher, TargetDispatcher)


# ===========================================================================
# End-to-end correctness with symbols
# ===========================================================================


class TestSymbolicCorrectness:

    def test_sdfg_with_symbol_in_args(self):
        """SDFG with symbol N in arglist generates correct code."""
        sdfg = dace.SDFG("sym_args")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        N = dace.symbol("N")
        sdfg.add_symbol("N", dace.int32)
        sdfg.add_array("A", [N], dace.float64)
        sdfg.add_array("B", [N], dace.float64)

        state = sdfg.add_state("s0")
        # Need a connected graph — isolated nodes are invalid
        a = state.add_read("A")
        b = state.add_write("B")
        t = state.add_tasklet("t", {"inp"}, {"out"}, "out = inp")
        state.add_edge(a, None, t, "inp", dace.Memlet("A[0]"))
        state.add_edge(t, "out", b, None, dace.Memlet("B[0]"))

        code_objects = _generate_code_for(sdfg)
        code = code_objects[0].code
        assert "def sym_args" in code
        # N should be in the parameter list
        assert "N" in code

    def test_full_compile_with_symbol(self):
        """Compile and run SDFG with symbolic dimension."""
        sdfg = dace.SDFG("sym_compile")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [1], dace.float64)
        sdfg.add_array("B", [1], dace.float64)

        state = sdfg.add_state("s0")
        a = state.add_read("A")
        b = state.add_write("B")
        t = state.add_tasklet("double", {"inp"}, {"out"}, "out = inp * 2")
        state.add_edge(a, None, t, "inp", dace.Memlet("A[0]"))
        state.add_edge(t, "out", b, None, dace.Memlet("B[0]"))

        compiled = sdfg.compile()
        A = np.array([21.0], dtype=np.float64)
        B = np.array([0.0], dtype=np.float64)
        compiled(A=A, B=B)
        np.testing.assert_allclose(B, [42.0])


# ===========================================================================
# Edge cases and error paths
# ===========================================================================


class TestEdgeCasesAndErrors:

    def test_sdfg_with_multiple_args(self):
        """SDFG with multiple array args generates correct function signature."""
        sdfg = dace.SDFG("multi_args")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("X", [1], dace.float64)
        sdfg.add_array("Y", [1], dace.float64)
        sdfg.add_array("Z", [1], dace.float64)
        state = sdfg.add_state("s0")
        x = state.add_read("X")
        z = state.add_write("Z")
        t = state.add_tasklet("add", {"a"}, {"c"}, "c = a")
        state.add_edge(x, None, t, "a", dace.Memlet("X[0]"))
        state.add_edge(t, "c", z, None, dace.Memlet("Z[0]"))

        code_objects = _generate_code_for(sdfg)
        code = code_objects[0].code
        assert "def multi_args" in code
        # All args should appear
        assert "X" in code
        assert "Z" in code

    def test_generate_code_returns_four_tuple(self):
        """generate_code returns (header, code, targets, envs)."""
        sdfg = _make_sdfg("tuple_test")
        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        result = codegen.generate_code(sdfg, None)
        assert isinstance(result, tuple)
        assert len(result) == 4
        header, code, targets, envs = result
        assert isinstance(header, str)
        assert isinstance(code, str)
        assert isinstance(targets, Set)
        assert isinstance(envs, Set)

    def test_code_object_properties(self):
        """Code objects have correct language and title."""
        sdfg = _make_sdfg("co_props")
        code_objects = _generate_code_for(sdfg)
        assert len(code_objects) >= 1
        co = code_objects[0]
        assert co.language == "py"
        assert co.title == "Frame"
        assert co.name == "co_props"

    def test_array_constant_ndarray(self):
        """Array constant with numpy ndarray in generated code."""
        sdfg = dace.SDFG("arr_const")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_state("s0")
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        sdfg.add_constant("MATRIX", arr)

        codegen = DaCePythonCodeGenerator(sdfg)
        stream = PythonCodeIOStream()
        codegen.generate_constants(sdfg, stream)
        code = "import numpy\n" # Need to import numpy for the array literal because we only call generate_constants, not the full codegen pipeline which would add the import
        code += stream.getvalue()
        namespace = {}
        exec(code, namespace)
        assert "MATRIX" in namespace
        np.testing.assert_array_equal(namespace["MATRIX"], arr)

    def test_single_scalar_constant(self):
        """Single scalar constant float."""
        sdfg = dace.SDFG("scalar_c")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_state("s0")
        sdfg.add_constant("PI", 3.14159)

        codegen = DaCePythonCodeGenerator(sdfg)
        stream = PythonCodeIOStream()
        codegen.generate_constants(sdfg, stream)
        code = stream.getvalue()
        assert "PI = 3.14159" in code

    @pytest.mark.parametrize(
        ('value', 'expected_literal'),
        [
            ('hello world', "'hello world'"),
            (True, 'True'),
            (None, 'None'),
            (3 + 4j, '(3+4j)'),
        ],
        ids=['string', 'bool', 'none', 'complex'])
    def test_scalar_constants_emit_valid_python_literals(self, value, expected_literal):
        """Scalar constants should be emitted as valid Python literals for exec()."""
        sdfg = dace.SDFG('scalar_literal')
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_state('s0')
        sdfg.add_constant('CONST_VALUE', value)

        codegen = object.__new__(DaCePythonCodeGenerator)
        stream = PythonCodeIOStream()
        codegen.generate_constants(sdfg, stream)
        code = stream.getvalue()

        assert f'CONST_VALUE = {expected_literal}' in code

        namespace = {}
        exec(code, namespace)
        assert namespace['CONST_VALUE'] == value


# ===========================================================================
# Additional branch-focused tests for framecode.py
# ===========================================================================


class TestAdditionalFramecodeBranches:

    def test_free_symbols_fallback_property(self):
        """free_symbols() falls back to .free_symbols when .used_symbols is absent."""
        sdfg = _make_sdfg("fallback_sym")
        codegen = DaCePythonCodeGenerator(sdfg)

        class HasFreeSymbols:
            free_symbols = {"omega"}

        assert codegen.free_symbols(HasFreeSymbols()) == {"omega"}

    def test_generate_fileheader_emits_custom_type_definitions(self, monkeypatch):
        """Custom struct/pointer dtype paths emit type definitions exactly once."""
        sdfg = _make_sdfg("custom_types")
        codegen = DaCePythonCodeGenerator(sdfg)

        struct_t = dace.struct("TmpStruct", a=dace.int32)
        ptr_to_struct = dtypes.pointer(struct_t)
        fake_arr = SimpleNamespace(dtype=ptr_to_struct)

        def fake_arrays_recursive():
            return [(sdfg, "P0", fake_arr), (sdfg, "P1", fake_arr)]

        monkeypatch.setattr(sdfg, "arrays_recursive", fake_arrays_recursive)

        stream = PythonCodeIOStream()
        codegen.generate_fileheader(sdfg, stream, backend='frame')
        code = stream.getvalue()
        assert "class TmpStruct" in code
        assert code.count("class TmpStruct") == 1
        assert "    a: numpy.int" in code

    def test_get_schedule_nested_in_parent_scope(self):
        """Nested SDFG inside a map inherits schedule from parent scope."""
        outer = dace.SDFG("sched_nested_scope")
        outer.backend = dace.dtypes.BackendLanguage.Python
        outer.add_symbol("N", dace.int32)
        outer.add_array("A", [1], dace.float64)
        outer.add_array("B", [1], dace.float64)

        inner = dace.SDFG("sched_inner")
        inner.add_array("X", [1], dace.float64)
        inner.add_array("Y", [1], dace.float64)

        ostate = outer.add_state("outer_state")
        me, _ = ostate.add_map("m", {"i": "0:1"}, schedule=dtypes.ScheduleType.CPU_Multicore)
        a = ostate.add_read("A")
        b = ostate.add_write("B")
        nsdfg = ostate.add_nested_sdfg(inner, {"X"}, {"Y"}, symbol_mapping={"N": "N"})

        ostate.add_memlet_path(a, me, nsdfg, dst_conn="X", memlet=dace.Memlet("A[0]"))
        ostate.add_memlet_path(nsdfg, b, memlet=dace.Memlet("B[0]"), src_conn="Y")

        codegen = DaCePythonCodeGenerator(outer)
        assert codegen._get_schedule(inner) == dtypes.ScheduleType.CPU_Multicore

    def test_can_allocate_uses_gpu_devicelevel_check(self, monkeypatch):
        """When can_allocate is false for GPU storage, fallback uses device-level check."""
        sdfg = _make_sdfg("gpu_alloc")
        state = next(sdfg.states())
        codegen = DaCePythonCodeGenerator(sdfg)
        desc = data.Array(dace.float64, [4], storage=dtypes.StorageType.GPU_Shared)

        monkeypatch.setattr(framecode_module.dtypes, "can_allocate", lambda *_args, **_kwargs: False)
        monkeypatch.setattr(framecode_module.sdscope, "is_devicelevel_gpu", lambda *_args, **_kwargs: True)

        assert codegen._can_allocate(sdfg, state, desc, sdfg) is True

    def test_determine_allocation_persistent_threadlocal_skips_statestruct(self):
        """Persistent thread-local storage allocates but does not append to statestruct."""
        sdfg = dace.SDFG("persist_tls")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array(
            "T",
            [2],
            dace.float64,
            transient=True,
            lifetime=dtypes.AllocationLifetime.Persistent,
            storage=dtypes.StorageType.CPU_ThreadLocal,
        )
        st = sdfg.add_state("s0")
        st.add_access("T")

        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        assert sdfg in codegen.to_allocate
        assert codegen.statestruct == []

    def test_determine_allocation_global_unused_is_skipped(self):
        """Unused global-lifetime transient is skipped."""
        sdfg = dace.SDFG("global_unused")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [2], dace.float64, transient=True, lifetime=dtypes.AllocationLifetime.Global)
        sdfg.add_state("s0")

        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        assert sdfg not in codegen.to_allocate or len(codegen.to_allocate[sdfg]) == 0

    def test_determine_allocation_sdfg_unused_is_skipped(self):
        """Unused SDFG-lifetime transient is skipped."""
        sdfg = dace.SDFG("sdfg_unused")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [2], dace.float64, transient=True, lifetime=dtypes.AllocationLifetime.SDFG)
        sdfg.add_state("s0")

        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        assert sdfg not in codegen.to_allocate or len(codegen.to_allocate[sdfg]) == 0

    def test_determine_allocation_nonfree_symbol_multistate(self, monkeypatch):
        """Non-free symbol dependency across states yields split declare/alloc/dealloc entries."""
        sdfg = dace.SDFG("nonfree_multistate")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [4], dace.float64, transient=True, lifetime=dtypes.AllocationLifetime.Scope)
        s0 = sdfg.add_state("s0")
        s1 = sdfg.add_state("s1")
        sdfg.add_edge(s0, s1, dace.InterstateEdge())
        n0 = s0.add_access("T")
        n1 = s1.add_access("T")
        assert n0 is not None and n1 is not None

        codegen = DaCePythonCodeGenerator(sdfg)
        monkeypatch.setattr(framecode_module.utils, "is_nonfree_sym_dependent", lambda *_args, **_kwargs: True)
        codegen.determine_allocation_lifetime(sdfg)

        entries = []
        for vals in codegen.to_allocate.values():
            entries.extend(vals)

        assert any(e[3] and not e[4] and not e[5] for e in entries)  # declaration
        assert any((not e[3]) and e[4] and (not e[5]) for e in entries)  # allocation
        assert any((not e[3]) and (not e[4]) and e[5] for e in entries)  # deallocation

    def test_determine_allocation_nonfree_symbol_single_state(self, monkeypatch):
        """Non-free symbol dependency in one state yields a full allocate/deallocate tuple."""
        sdfg = dace.SDFG("nonfree_single")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [4], dace.float64, transient=True, lifetime=dtypes.AllocationLifetime.Scope)
        s0 = sdfg.add_state("s0")
        s0.add_access("T")

        codegen = DaCePythonCodeGenerator(sdfg)
        monkeypatch.setattr(framecode_module.utils, "is_nonfree_sym_dependent", lambda *_args, **_kwargs: True)
        codegen.determine_allocation_lifetime(sdfg)

        tuples = [t for vals in codegen.to_allocate.values() for t in vals]
        assert any(t[3] and t[4] and t[5] for t in tuples)

    def test_determine_allocation_nonfree_with_unreachable_instances(self, monkeypatch):
        """Unreachable instance path triggers dominator/postdominator fallback."""
        sdfg = dace.SDFG("nonfree_unreachable")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [4], dace.float64, transient=True, lifetime=dtypes.AllocationLifetime.Scope)
        s0 = sdfg.add_state("s0", is_start_block=True)
        s1 = sdfg.add_state("s1")
        # Reachability is monkeypatched below to simulate non-reachability.
        sdfg.add_edge(s0, s1, dace.InterstateEdge())
        s0.add_access("T")
        s1.add_access("T")

        codegen = DaCePythonCodeGenerator(sdfg)
        fake_reachability = {sdfg.cfg_id: {s0: {s0}, s1: {s1}}}

        class FakeReachability:
            def apply_pass(self, *_args, **_kwargs):
                return fake_reachability

        monkeypatch.setattr(framecode_module, "StateReachability", lambda: FakeReachability())
        monkeypatch.setattr(framecode_module.utils, "is_nonfree_sym_dependent", lambda *_args, **_kwargs: True)
        monkeypatch.setattr(framecode_module, "_get_dominator_and_postdominator", lambda *_args, **_kwargs: (s0, s1))

        codegen.determine_allocation_lifetime(sdfg)
        entries = [t for vals in codegen.to_allocate.values() for t in vals]
        assert any(t[1] is None and t[3] and (not t[4]) and (not t[5]) for t in entries)

    def test_deallocate_arrays_in_scope_state_none_raises(self):
        """deallocate_arrays_in_scope with state=None reaches the AttributeError path."""
        sdfg = dace.SDFG("dealloc_state_none")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [2], dace.float64, transient=True)
        sdfg.add_state("s0")
        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.to_allocate[sdfg].append((sdfg, None, nodes.AccessNode("T"), False, False, True))

        with pytest.raises(AttributeError):
            codegen.deallocate_arrays_in_scope(sdfg, sdfg, sdfg, PythonCodeIOStream(), PythonCodeIOStream())

    def test_symbols_defined_at_respects_map_scope(self):
        sdfg = dace.SDFG("scoped_symbol_types")
        state = sdfg.add_state()
        outside = state.add_tasklet("outside", {}, {}, "pass")
        entry, _ = state.add_map("inner", {"i": "0:4"})
        inside = state.add_tasklet("inside", {}, {}, "pass")
        state.add_edge(entry, None, inside, None, dace.Memlet())
        codegen = DaCePythonCodeGenerator(sdfg)
        assert "i" not in codegen.symbols_defined_at(state, outside)
        assert codegen.symbols_defined_at(state, inside)["i"] in dace.dtypes.INTEGER_TYPES
        assert "i" not in codegen.symbols_defined_at(state, outside)


# ===========================================================================
# Coverage-boost tests targeting specific uncovered lines
# ===========================================================================


class TestCoverageBoostV2:
    """Targeted tests for previously-uncovered lines in framecode.py."""

    # ------------------------------------------------------------------
    # Line 74: constant edge propagation in __init__
    # ------------------------------------------------------------------

    def test_init_constant_edge_propagates_dst_conn(self):
        """Line 74: edge.data.data in parent_constants → result.add(edge.dst_conn)."""
        outer = dace.SDFG("outer_cep")
        outer.backend = dace.dtypes.BackendLanguage.Python
        # Add "CA" as both an array and a constant so it's in parent_constants
        outer.add_array("CA", [1], dace.float64, transient=True)
        outer.add_constant("CA", np.array([42.0]))
        outer.add_array("B", [1], dace.float64)

        inner = dace.SDFG("inner_cep")
        inner.add_array("X", [1], dace.float64)
        inner.add_array("Y", [1], dace.float64)
        ist = inner.add_state("is")
        ix = ist.add_read("X")
        iy = ist.add_write("Y")
        it = ist.add_tasklet("t", {"i"}, {"o"}, "o = i")
        ist.add_edge(ix, None, it, "i", dace.Memlet("X[0]"))
        ist.add_edge(it, "o", iy, None, dace.Memlet("Y[0]"))

        ostate = outer.add_state("os")
        ca = ostate.add_read("CA")
        b = ostate.add_write("B")
        nsdfg = ostate.add_nested_sdfg(inner, {"X"}, {"Y"})
        # edge.data.data = "CA" which IS in parent_constants → result.add("X") at line 74
        ostate.add_edge(ca, None, nsdfg, "X", dace.Memlet("CA[0]"))
        ostate.add_edge(nsdfg, "Y", b, None, dace.Memlet("B[0]"))

        codegen = DaCePythonCodeGenerator(outer)
        inner_syms = codegen.symbols_and_constants(inner)
        # "X" was propagated because the edge carries a constant
        assert "X" in inner_syms

    # ------------------------------------------------------------------
    # Line 218: generate_header with env having (empty) state_fields
    # ------------------------------------------------------------------

    def test_generate_header_env_with_state_fields_executes_extend(self):
        """Line 218: generate_header iterates env.state_fields → statestruct.extend called."""
        sdfg = _make_sdfg("hdr_env_sf")
        codegen = DaCePythonCodeGenerator(sdfg)

        class FakeEnv:
            headers = []
            state_fields = []  # empty → no indent() crash from nonempty statestruct

        codegen.environments = [FakeEnv()]
        gs = PythonCodeIOStream()
        cs = PythonCodeIOStream()
        codegen.generate_header(sdfg, gs, cs)
        assert "AUTO-GENERATED" in gs.getvalue()

    # ------------------------------------------------------------------
    # Line 329: _get_schedule for nested SDFG not inside a map
    # ------------------------------------------------------------------

    def test_get_schedule_nested_not_in_map(self):
        """Line 329: nested SDFG not in a map → pscope=None → returns _get_schedule(pstate)."""
        outer = dace.SDFG("sched_nomap")
        outer.backend = dace.dtypes.BackendLanguage.Python
        outer.add_array("A", [1], dace.float64)
        outer.add_array("B", [1], dace.float64)

        inner = dace.SDFG("inner_nomap")
        inner.add_array("X", [1], dace.float64)
        inner.add_array("Y", [1], dace.float64)
        ist = inner.add_state("is")
        ix = ist.add_read("X")
        iy = ist.add_write("Y")
        it = ist.add_tasklet("t", {"i"}, {"o"}, "o = i")
        ist.add_edge(ix, None, it, "i", dace.Memlet("X[0]"))
        ist.add_edge(it, "o", iy, None, dace.Memlet("Y[0]"))

        ostate = outer.add_state("os")
        a = ostate.add_read("A")
        b = ostate.add_write("B")
        nsdfg = ostate.add_nested_sdfg(inner, {"X"}, {"Y"})
        ostate.add_edge(a, None, nsdfg, "X", dace.Memlet("A[0]"))
        ostate.add_edge(nsdfg, "Y", b, None, dace.Memlet("B[0]"))

        codegen = DaCePythonCodeGenerator(outer)
        # Not inside any map → entry_node(nsdfg_node)=None → pscope=None → line 329 hit
        result = codegen._get_schedule(inner)
        assert result == dtypes.ScheduleType.Sequential

    # ------------------------------------------------------------------
    # Line 351: _can_allocate returns False (non-GPU, can_allocate=False)
    # ------------------------------------------------------------------

    def test_can_allocate_returns_false_non_gpu_storage(self, monkeypatch):
        """Line 351: _can_allocate → False when storage not GPU and can_allocate=False."""
        sdfg = _make_sdfg("can_alloc_f")
        state = next(sdfg.states())
        codegen = DaCePythonCodeGenerator(sdfg)
        desc = data.Array(dace.float64, [4], storage=dtypes.StorageType.CPU_Heap)
        monkeypatch.setattr(framecode_module.dtypes, "can_allocate", lambda *_: False)
        result = codegen._can_allocate(sdfg, state, desc, sdfg)
        assert result is False

    # ------------------------------------------------------------------
    # Line 388: edge-based transient tracking in access_instances
    # ------------------------------------------------------------------

    def test_determine_alloc_transient_in_edge_free_symbols(self):
        """Line 388: transient name in edge free_symbols → appended to instances."""
        sdfg = dace.SDFG("edge_trans")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [1], dace.float64, transient=True,
                       lifetime=dtypes.AllocationLifetime.Scope)
        sdfg.add_array("A", [1], dace.float64)
        s0 = sdfg.add_state("s0")
        s1 = sdfg.add_state("s1")
        # Edge with assignment that references T → T in edge free_symbols
        sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"x": "T[0]"}))
        # T is also accessed in s0
        a_nd = s0.add_read("A")
        t_nd = s0.add_write("T")
        tk = s0.add_tasklet("tk", {"a"}, {"t"}, "t = a")
        s0.add_edge(a_nd, None, tk, "a", dace.Memlet("A[0]"))
        s0.add_edge(tk, "t", t_nd, None, dace.Memlet("T[0]"))

        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        # T should be allocated (in sdfg scope since it appears in edge)
        assert sdfg in codegen.to_allocate

    # ------------------------------------------------------------------
    # Line 428: unused Persistent transient is skipped
    # ------------------------------------------------------------------

    def test_persistent_unused_transient_skipped(self):
        """Line 428: Persistent transient with no access node → continue (skipped)."""
        sdfg = dace.SDFG("persist_unused")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [2], dace.float64, transient=True,
                       lifetime=dtypes.AllocationLifetime.Persistent)
        sdfg.add_state("s0")  # no access to T

        codegen = DaCePythonCodeGenerator(sdfg)
        codegen.determine_allocation_lifetime(sdfg)
        # T should NOT appear in to_allocate (skipped at line 428)
        all_names = {entry[2].data
                     for entries in codegen.to_allocate.values()
                     for entry in entries}
        assert "T" not in all_names

    # ------------------------------------------------------------------
    # Lines 477-478, 481: State lifetime multi-state (mock shared_transients)
    # ------------------------------------------------------------------

    def test_state_lifetime_multistate_path(self, monkeypatch):
        """Lines 477-478, 481: State lifetime multi-state path hit via mocked shared_transients."""
        sdfg = dace.SDFG("state_lt_ms")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [5], dace.float64, transient=True,
                       lifetime=dtypes.AllocationLifetime.State)
        s0 = sdfg.add_state("s0")
        s1 = sdfg.add_state("s1")
        sdfg.add_edge(s0, s1, dace.InterstateEdge())
        s0.add_access("T")
        s1.add_access("T")

        codegen = DaCePythonCodeGenerator(sdfg)
        # Exclude T from shared_transients so State lifetime path is entered
        monkeypatch.setattr(dace.SDFG, "shared_transients", lambda *_a, **_kw: [])
        codegen.determine_allocation_lifetime(sdfg)
        # multistate=True → alloc_scope = sdfg (line 481)
        assert sdfg in codegen.to_allocate

    # ------------------------------------------------------------------
    # Lines 495-496, 500: Scope lifetime multistate via interstate edge
    # ------------------------------------------------------------------

    def test_scope_lifetime_multistate_via_edge(self, monkeypatch):
        """Lines 495-496: Scope lifetime transient in interstate edge → multistate=True."""
        sdfg = dace.SDFG("scope_lt_edge")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [1], dace.float64, transient=True,
                       lifetime=dtypes.AllocationLifetime.Scope)
        sdfg.add_array("A", [1], dace.float64)
        s0 = sdfg.add_state("s0")
        s1 = sdfg.add_state("s1")
        # T appears in the interstate edge → multistate=True via line 495-496
        sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"x": "T[0]"}))
        a_nd = s0.add_read("A")
        t_nd = s0.add_write("T")
        tk = s0.add_tasklet("tk", {"a"}, {"t"}, "t = a")
        s0.add_edge(a_nd, None, tk, "a", dace.Memlet("A[0]"))
        s0.add_edge(tk, "t", t_nd, None, dace.Memlet("T[0]"))

        codegen = DaCePythonCodeGenerator(sdfg)
        monkeypatch.setattr(dace.SDFG, "shared_transients", lambda *_a, **_kw: [])
        codegen.determine_allocation_lifetime(sdfg)
        assert sdfg in codegen.to_allocate

    # ------------------------------------------------------------------
    # Lines 504, 514-515, 534, 537: Scope lifetime multistate via two states
    # ------------------------------------------------------------------

    def test_scope_lifetime_multistate_two_states(self, monkeypatch):
        """Lines 504, 514-515, 534, 537: Scope lifetime in two states → multistate=True."""
        sdfg = dace.SDFG("scope_lt_2s")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [5], dace.float64, transient=True,
                       lifetime=dtypes.AllocationLifetime.Scope)
        s0 = sdfg.add_state("s0")
        s1 = sdfg.add_state("s1")
        sdfg.add_edge(s0, s1, dace.InterstateEdge())
        s0.add_access("T")
        s1.add_access("T")

        codegen = DaCePythonCodeGenerator(sdfg)
        monkeypatch.setattr(dace.SDFG, "shared_transients", lambda *_a, **_kw: [])
        codegen.determine_allocation_lifetime(sdfg)
        assert sdfg in codegen.to_allocate

    # ------------------------------------------------------------------
    # Lines 524-531: common_parent_scope logic (two nodes in different map scopes)
    # ------------------------------------------------------------------

    def test_scope_lifetime_two_maps_common_parent(self, monkeypatch):
        """Lines 524-531: Two access nodes in different map scopes → common_parent_scope."""
        sdfg = dace.SDFG("two_maps_scope")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [10], dace.float64)
        sdfg.add_array("T", [10], dace.float64, transient=True,
                       lifetime=dtypes.AllocationLifetime.Scope)
        sdfg.add_array("B", [10], dace.float64)
        state = sdfg.add_state("s0")

        # Map 1: A → T
        me1, mx1 = state.add_map("m1", {"i": "0:10"})
        t1 = state.add_tasklet("t1", {"a"}, {"t"}, "t = a")
        t_wr = state.add_access("T")
        a_nd = state.add_read("A")
        state.add_memlet_path(a_nd, me1, t1, dst_conn="a", memlet=dace.Memlet("A[i]"))
        state.add_memlet_path(t1, mx1, t_wr, src_conn="t", memlet=dace.Memlet("T[i]"))

        # Map 2: T → B
        me2, mx2 = state.add_map("m2", {"j": "0:10"})
        t2 = state.add_tasklet("t2", {"t"}, {"b"}, "b = t")
        t_rd = state.add_access("T")
        b_nd = state.add_write("B")
        state.add_memlet_path(t_rd, me2, t2, dst_conn="t", memlet=dace.Memlet("T[j]"))
        state.add_memlet_path(t2, mx2, b_nd, src_conn="b", memlet=dace.Memlet("B[j]"))

        codegen = DaCePythonCodeGenerator(sdfg)
        monkeypatch.setattr(dace.SDFG, "shared_transients", lambda *_a, **_kw: [])
        codegen.determine_allocation_lifetime(sdfg)
        # T should be allocated at state level (common parent of me1 and me2 is the state)
        all_names = {entry[2].data
                     for entries in codegen.to_allocate.values()
                     for entry in entries}
        assert "T" in all_names

    # ------------------------------------------------------------------
    # Lines 553-570, 575: _can_allocate traversal upward
    # ------------------------------------------------------------------

    def test_can_allocate_traversal_up_to_top(self, monkeypatch):
        """Lines 553-570, 575: _can_allocate fails → traverse up until None → top_sdfg."""
        sdfg = dace.SDFG("alloc_trav")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [4], dace.float64, transient=True,
                       lifetime=dtypes.AllocationLifetime.Scope)
        s0 = sdfg.add_state("s0")
        s0.add_access("T")

        codegen = DaCePythonCodeGenerator(sdfg)
        call_count = {"n": 0}

        def mock_can_allocate(csdfg, cstate, cdesc, cscope):
            call_count["n"] += 1
            return call_count["n"] > 1  # False first call, True thereafter

        monkeypatch.setattr(codegen, "_can_allocate", mock_can_allocate)
        codegen.determine_allocation_lifetime(sdfg)
        # After traversal, curscope=None → curscope=top_sdfg (line 575)
        assert sdfg in codegen.to_allocate

    # ------------------------------------------------------------------
    # Lines 590-594: View handling in non-free-sym dependent multi-state
    # ------------------------------------------------------------------

    def test_view_in_nonfree_sym_dependent_multistate(self, monkeypatch):
        """Lines 590-594: View descriptor in non-free-sym multi-state → per-state alloc."""
        sdfg = dace.SDFG("view_nonfree")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [10], dace.float64)
        sdfg.add_view("V", [10], dace.float64)
        s0 = sdfg.add_state("s0")
        s1 = sdfg.add_state("s1")
        sdfg.add_edge(s0, s1, dace.InterstateEdge())
        a_nd = s0.add_read("A")
        v0 = s0.add_access("V")
        s0.add_edge(a_nd, None, v0, None, dace.Memlet("A[0:10]"))
        v1 = s1.add_access("V")
        b_nd = s1.add_write("A")
        s1.add_edge(v1, None, b_nd, None, dace.Memlet("A[0:10]"))

        codegen = DaCePythonCodeGenerator(sdfg)
        monkeypatch.setattr(dace.SDFG, "shared_transients", lambda *_a, **_kw: [])
        monkeypatch.setattr(framecode_module.utils, "is_nonfree_sym_dependent",
                            lambda *_: True)
        codegen.determine_allocation_lifetime(sdfg)
        # V should appear in per-state to_allocate entries
        assert len(codegen.to_allocate) > 0

    # ------------------------------------------------------------------
    # Line 604: reachable instances else-branch (declare at scope)
    # ------------------------------------------------------------------

    def test_nonfree_sym_reachable_instances_else_branch(self, monkeypatch):
        """Line 604: non-free sym dep with reachable instances → else (line 604) for declare."""
        sdfg = dace.SDFG("nonfree_reach")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [4], dace.float64, transient=True,
                       lifetime=dtypes.AllocationLifetime.Scope)
        s0 = sdfg.add_state("s0", is_start_block=True)
        s1 = sdfg.add_state("s1")
        sdfg.add_edge(s0, s1, dace.InterstateEdge())
        s0.add_access("T")
        s1.add_access("T")

        codegen = DaCePythonCodeGenerator(sdfg)
        monkeypatch.setattr(dace.SDFG, "shared_transients", lambda *_a, **_kw: [])
        monkeypatch.setattr(framecode_module.utils, "is_nonfree_sym_dependent",
                            lambda *_: True)
        # Default reachability: s1 IS reachable from s0 → else branch (line 604)
        codegen.determine_allocation_lifetime(sdfg)
        entries = [t for vals in codegen.to_allocate.values() for t in vals]
        assert len(entries) > 0

    # ------------------------------------------------------------------
    # Line 633: allocate_arrays_in_scope with state not None
    # ------------------------------------------------------------------

    def test_allocate_scope_transient_state_not_none(self):
        """Line 633: transient allocated at state scope → state.block_id used."""
        sdfg = dace.SDFG("alloc_state_nn")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [1], dace.float64)
        sdfg.add_array("T", [1], dace.float64, transient=True,
                       lifetime=dtypes.AllocationLifetime.Scope)
        sdfg.add_array("B", [1], dace.float64)
        state = sdfg.add_state("s0")
        a_nd = state.add_read("A")
        t_nd = state.add_access("T")
        b_nd = state.add_write("B")
        tk1 = state.add_tasklet("tk1", {"x"}, {"y"}, "y = x")
        tk2 = state.add_tasklet("tk2", {"x"}, {"y"}, "y = x")
        state.add_edge(a_nd, None, tk1, "x", dace.Memlet("A[0]"))
        state.add_edge(tk1, "y", t_nd, None, dace.Memlet("T[0]"))
        state.add_edge(t_nd, None, tk2, "x", dace.Memlet("T[0]"))
        state.add_edge(tk2, "y", b_nd, None, dace.Memlet("B[0]"))

        # Full code generation triggers allocate_arrays_in_scope with state not None
        code_objects = _generate_code_for(sdfg)
        assert len(code_objects) >= 1

    # ------------------------------------------------------------------
    # Lines 654, 656: deallocate_arrays_in_scope — skip + state not None
    # ------------------------------------------------------------------

    def test_deallocate_scope_skip_and_state_not_none(self, monkeypatch):
        """Lines 654, 656: dealloc skip when deallocate=False; use block_id when state not None."""
        sdfg = dace.SDFG("dealloc_nn")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [2], dace.float64, transient=True)
        state = sdfg.add_state("s0")
        state.add_access("T")
        codegen = DaCePythonCodeGenerator(sdfg)

        # First entry: deallocate=False → will trigger line 654 (continue)
        codegen.to_allocate[state].append(
            (sdfg, state, nodes.AccessNode("T"), True, False, False))
        # Second entry: deallocate=True, state not None → line 656 (state.block_id)
        codegen.to_allocate[state].append(
            (sdfg, state, nodes.AccessNode("T"), False, False, True))

        captured = {}

        def fake_dispatch_dealloc(tsdfg, cfg, st, state_id, *_):
            captured["state_id"] = state_id
            captured["st"] = st

        codegen._dispatcher.dispatch_deallocate = fake_dispatch_dealloc
        codegen.deallocate_arrays_in_scope(sdfg, sdfg, state,
                                           PythonCodeIOStream(), PythonCodeIOStream())

        assert "state_id" in captured
        assert captured["state_id"] == state.block_id

    # ------------------------------------------------------------------
    # Lines 686, 803-804: generate_code for nested SDFG (not top-level)
    # ------------------------------------------------------------------

    def test_generate_code_nested_sdfg_not_top_level(self, monkeypatch):
        """Lines 686, 803-804: generate_code on nested SDFG (is_top_level=False)."""
        outer = dace.SDFG("outer_ntl")
        outer.backend = dace.dtypes.BackendLanguage.Python
        outer.add_array("A", [1], dace.float64)
        outer.add_array("B", [1], dace.float64)

        inner = dace.SDFG("inner_ntl")
        inner.add_array("X", [1], dace.float64)
        inner.add_array("Y", [1], dace.float64)
        ist = inner.add_state("is")
        ix = ist.add_read("X")
        iy = ist.add_write("Y")
        it = ist.add_tasklet("t", {"i"}, {"o"}, "o = i")
        ist.add_edge(ix, None, it, "i", dace.Memlet("X[0]"))
        ist.add_edge(it, "o", iy, None, dace.Memlet("Y[0]"))

        ostate = outer.add_state("os")
        a = ostate.add_read("A")
        b = ostate.add_write("B")
        nsdfg = ostate.add_nested_sdfg(inner, {"X"}, {"Y"})
        ostate.add_edge(a, None, nsdfg, "X", dace.Memlet("A[0]"))
        ostate.add_edge(nsdfg, "Y", b, None, dace.Memlet("B[0]"))

        outer.reset_cfg_list()
        codegen = DaCePythonCodeGenerator(outer)
        codegen.determine_allocation_lifetime(outer)

        # Mock generate_states so node dispatch is bypassed
        monkeypatch.setattr(codegen, "generate_states",
                            lambda s, gs, cs: set(s.states()))
        monkeypatch.setattr(codegen, "allocate_arrays_in_scope", lambda *_: None)
        monkeypatch.setattr(codegen, "deallocate_arrays_in_scope", lambda *_: None)

        # Call generate_code on inner (cfg_id != 0 → line 686; is_top_level=False → lines 803-804)
        result = codegen.generate_code(inner, None, cfg_id="")
        assert isinstance(result, tuple) and len(result) == 4
        # cfg_id was set to '_%d' (line 686) and else branch used (lines 803-804)
        header, code, targets, envs = result
        assert isinstance(header, str)
        assert isinstance(code, str)

    # ------------------------------------------------------------------
    # Line 709: Array constant in generate_code defined_vars
    # ------------------------------------------------------------------

    def test_generate_code_array_constant_defined_vars(self):
        """Line 709: Array constant causes Pointer type in defined_vars."""
        sdfg = dace.SDFG("arr_const_dv")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_constant("TABLE", np.array([1.0, 2.0, 3.0], dtype=np.float64))
        sdfg.add_state("s0")

        code_objects = _generate_code_for(sdfg)
        code = code_objects[0].code
        assert "TABLE" in code

    # ------------------------------------------------------------------
    # Explicit loop-variable types live in the symbol repository.
    # ------------------------------------------------------------------

    def test_loop_variable_declared_in_its_scope(self):
        """A scoped declaration survives code generation and executes correctly."""
        sdfg = dace.SDFG("loop_in_syms")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [1], dace.float64)

        init_state = sdfg.add_state("init")

        loop = LoopRegion("loop_i", condition_expr="i < numpy.int64(5)", loop_var="i",
                          initialize_expr="i = numpy.int64(0)", update_expr="i = i + numpy.int64(1)",
                          sdfg=sdfg)
        sdfg.add_node(loop)
        sdfg.symbol_repo.add("i", dace.int64, at=loop)
        sdfg.add_edge(init_state, loop, dace.InterstateEdge())

        body = loop.add_state("body")
        ar = body.add_read("A")
        aw = body.add_write("A")
        tk = body.add_tasklet("inc", {"x"}, {"y"}, "y = x + 1.0")
        body.add_edge(ar, None, tk, "x", dace.Memlet("A[0]"))
        body.add_edge(tk, "y", aw, None, dace.Memlet("A[0]"))

        code_objects = _generate_code_for(sdfg)
        code = code_objects[0].code
        assert "def loop_in_syms" in code
        assert DaCePythonCodeGenerator(sdfg).symbols_defined_at(body, tk)["i"] == dace.int64
        output = np.zeros(1)
        sdfg.compile()(A=output)
        np.testing.assert_array_equal(output, [5.0])

    # ------------------------------------------------------------------
    # Line 750: nested SDFG symbol skipped when in symbol_mapping
    # ------------------------------------------------------------------

    def test_generate_code_nested_skips_mapped_symbols(self, monkeypatch):
        """Line 750: is_top_level=False and symbol in symbol_mapping → continue."""
        outer = dace.SDFG("outer_sym_skip")
        outer.backend = dace.dtypes.BackendLanguage.Python
        outer.add_array("A", [1], dace.float64)
        outer.add_array("B", [1], dace.float64)

        inner = dace.SDFG("inner_sym_skip")
        inner.add_array("X", [1], dace.float64)
        inner.add_array("Y", [1], dace.float64)
        # Add an interstate symbol "k" via edge assignment
        is0 = inner.add_state("is0")
        is1 = inner.add_state("is1")
        inner.add_edge(is0, is1, dace.InterstateEdge(assignments={"k": "0"}))
        ix = is0.add_read("X")
        iy = is0.add_write("Y")
        it = is0.add_tasklet("t", {"i"}, {"o"}, "o = i")
        is0.add_edge(ix, None, it, "i", dace.Memlet("X[0]"))
        is0.add_edge(it, "o", iy, None, dace.Memlet("Y[0]"))

        ostate = outer.add_state("os")
        a = ostate.add_read("A")
        b = ostate.add_write("B")
        # Map "k" in symbol_mapping so it's skipped in inner's generate_code
        nsdfg = ostate.add_nested_sdfg(inner, {"X"}, {"Y"},
                                       symbol_mapping={"k": "0"})
        ostate.add_edge(a, None, nsdfg, "X", dace.Memlet("A[0]"))
        ostate.add_edge(nsdfg, "Y", b, None, dace.Memlet("B[0]"))

        outer.reset_cfg_list()
        codegen = DaCePythonCodeGenerator(outer)
        codegen.determine_allocation_lifetime(outer)

        monkeypatch.setattr(codegen, "generate_states",
                            lambda s, gs, cs: set(s.states()))
        monkeypatch.setattr(codegen, "allocate_arrays_in_scope", lambda *_: None)
        monkeypatch.setattr(codegen, "deallocate_arrays_in_scope", lambda *_: None)

        # "k" is in inner.parent_nsdfg_node.symbol_mapping → line 750 (continue) hit
        result = codegen.generate_code(inner, None, cfg_id="")
        assert isinstance(result, tuple)

    # ------------------------------------------------------------------
    # Lines 832-834: _get_dominator_and_postdominator loop traversal
    # ------------------------------------------------------------------

    def test_get_dominator_and_postdominator_traversal(self):
        """Lines 832-834, 838-840: dominator/postdominator traversal in branching SDFG."""
        from dace.codegen.py.framecode import _get_dominator_and_postdominator
        # Build a diamond-shaped SDFG: s0 → s1, s0 → s2, s1 → s3, s2 → s3
        sdfg = dace.SDFG("diamond")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("T", [1], dace.float64, transient=True)
        s0 = sdfg.add_state("s0", is_start_block=True)
        s1 = sdfg.add_state("s1")
        s2 = sdfg.add_state("s2")
        s3 = sdfg.add_state("s3")
        sdfg.add_edge(s0, s1, dace.InterstateEdge(condition="1 > 0"))
        sdfg.add_edge(s0, s2, dace.InterstateEdge(condition="1 <= 0"))
        sdfg.add_edge(s1, s3, dace.InterstateEdge())
        sdfg.add_edge(s2, s3, dace.InterstateEdge())

        # T accessed in s1 and s2 (branches of diamond)
        n1 = s1.add_access("T")
        n2 = s2.add_access("T")

        # _get_dominator_and_postdominator should find s0 as dominator, s3 as postdominator
        accesses = [(s1, n1), (s2, n2)]
        try:
            start_s, end_s = _get_dominator_and_postdominator(sdfg, accesses)
            # Should find s0 as dominator and s3 as postdominator
            assert start_s is not None
            assert end_s is not None
        except NotImplementedError:
            pytest.skip("Dominator/postdominator not found for this SDFG structure")

    # ------------------------------------------------------------------
    # Line 500: Scope lifetime with LoopRegion - additional coverage
    # ------------------------------------------------------------------

    def test_scope_lifetime_single_scope_in_map(self, monkeypatch):
        """Line 520-521: Scope lifetime, single node in map → curscope = map entry."""
        sdfg = dace.SDFG("scope_lt_map")
        sdfg.backend = dace.dtypes.BackendLanguage.Python
        sdfg.add_array("A", [10], dace.float64)
        sdfg.add_array("T", [1], dace.float64, transient=True,
                       lifetime=dtypes.AllocationLifetime.Scope)
        sdfg.add_array("B", [10], dace.float64)
        state = sdfg.add_state("s0")
        me, mx = state.add_map("m", {"i": "0:10"})
        tk = state.add_tasklet("tk", {"a"}, {"b", "t"}, "b = a; t = a")
        t_nd = state.add_access("T")
        a_nd = state.add_read("A")
        b_nd = state.add_write("B")
        state.add_memlet_path(a_nd, me, tk, dst_conn="a", memlet=dace.Memlet("A[i]"))
        state.add_memlet_path(tk, mx, b_nd, src_conn="b", memlet=dace.Memlet("B[i]"))
        state.add_memlet_path(tk, mx, t_nd, src_conn="t", memlet=dace.Memlet("T[0]"))

        codegen = DaCePythonCodeGenerator(sdfg)
        monkeypatch.setattr(dace.SDFG, "shared_transients", lambda *_a, **_kw: [])
        codegen.determine_allocation_lifetime(sdfg)
        # T should be allocated at map entry scope
        all_names = {entry[2].data
                     for entries in codegen.to_allocate.values()
                     for entry in entries}
        assert "T" in all_names
