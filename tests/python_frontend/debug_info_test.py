from dace import SDFG, SDFGState
from dace import dtypes
from dace.config import Config, temporary_config

import pytest


@pytest.fixture()
def state() -> SDFGState:
    """Setup a simple SDFG for testing and return the state."""

    sdfg = SDFG("tester")
    sdfg.add_array("A", [10], dtypes.int64)
    sdfg.add_array("B", [15], dtypes.int64)

    return sdfg.add_state("start_here", is_start_block=True)


def test_config_compiler_lineinfo_none(state: SDFGState) -> None:
    with temporary_config():
        Config.set(*["compiler", "lineinfo"], value="none")
        node = state.add_access("A")
        assert node.debuginfo is None


def test_config_compiler_lineinfo_inspect(state: SDFGState) -> None:
    # Ensure "inspect" is the default configuration
    assert Config.get(*["compiler", "lineinfo"]) == "inspect"

    # Add an access and expect debug info to be added (from this file)
    node = state.add_access("A")
    assert node.debuginfo is not None
    assert node.debuginfo.filename == __file__


def test_using_default_lineinfo(state: SDFGState) -> None:
    line = 42
    filename = "my/test/path.py"
    state._default_lineinfo = dtypes.DebugInfo(start_line=line, filename=filename)

    read_A = state.add_read("A")
    assert read_A.debuginfo is not None
    assert read_A.debuginfo.filename == filename
    assert read_A.debuginfo.start_line == line

    with pytest.deprecated_call():
        read_B = state.add_read("B", None)
    assert read_B.debuginfo is not None
    assert read_B.debuginfo.filename == filename
    assert read_B.debuginfo.start_line == line


def test_passing_debug_info_warns(state: SDFGState) -> None:
    with pytest.deprecated_call():
        state.add_access("A", debuginfo=dtypes.DebugInfo(start_line=42))


def test_passing_None_warns(state: SDFGState) -> None:
    with pytest.deprecated_call():
        state.add_access("A", debuginfo=None)


if __name__ == "__main__":
    test_config_compiler_lineinfo_none()
    test_config_compiler_lineinfo_inspect()
    test_using_default_lineinfo()
    test_passing_debug_info_warns()
    test_passing_None_warns()
