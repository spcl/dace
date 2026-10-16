# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for library environments requested by the model and materialized on disk."""

import os
import pathlib
import tempfile

import pytest

import dace
import dace.library
from dace import nodes
from dace.libraries.ai import environments as ai_environments
from dace.libraries.ai.backend import EnvironmentSpec, TaskletSpec
from dace.libraries.ai.context import collect_context
from dace.libraries.ai.exceptions import AIExpansionError
from dace.libraries.ai.nodes import AINode
from dace.libraries.ai.prompts import build_user_prompt
from tests.library.ai.ai_test_utils import offline, stub_provider, stub_provider_sequence

SPEC = EnvironmentSpec(
    name="TestFFT",
    headers=["fftw3.h"],
    cmake_packages=["PkgConfig"],
    cmake_libraries=["fftw3"],
    cmake_compile_flags=["-DTEST_FFT=1"],
    cmake_minimum_version="3.15",
    state_fields=["int test_fft_plan;"],
    init_code="// set up",
    finalize_code="// tear down",
)


def test_render_produces_every_required_field():
    source = ai_environments.render_environment(SPEC)

    # dace.library.environment enforces all thirteen fields
    for field in (
        "cmake_minimum_version",
        "cmake_packages",
        "cmake_variables",
        "cmake_includes",
        "cmake_libraries",
        "cmake_compile_flags",
        "cmake_link_flags",
        "cmake_files",
        "headers",
        "state_fields",
        "init_code",
        "finalize_code",
        "dependencies",
    ):
        assert f"    {field} = " in source, f"{field} is missing from the generated module"
    assert "class TestFFT:" in source
    assert "'fftw3.h'" in source


def test_materialize_writes_and_registers(tmp_path):
    with dace.config.set_temporary("default_build_folder", value=str(tmp_path)):
        env = ai_environments.materialize(SPEC)

        path = tmp_path / "ai-environments" / "dace_ai_env_testfft.py"
        assert path.exists(), "the environment was not written to disk"

        assert env._dace_library_environment
        assert env.cmake_libraries == ["fftw3"]
        assert env.state_fields == ["int test_fft_plan;"]
        # Registered under its full class path, which is what expansions store on the node
        assert dace.library.get_environment(env.full_class_path()) is env
        # The path is resolvable, which is what @dace.library.environment needs it for
        assert os.path.exists(env._dace_file_path)


def test_generated_environments_reload(tmp_path):
    with dace.config.set_temporary("default_build_folder", value=str(tmp_path)):
        ai_environments.materialize(SPEC)
        # Forget the registration, as a fresh interpreter would
        registry = dace.library._DACE_REGISTERED_ENVIRONMENTS
        stale = [k for k in registry if k.startswith("dace_ai_env_testfft")]
        for key in stale:
            del registry[key]

        loaded = ai_environments.load_generated_environments()

    assert any(env.__name__ == "TestFFT" for env in loaded)
    assert any(k.startswith("dace_ai_env_testfft") for k in dace.library._DACE_REGISTERED_ENVIRONMENTS)


def test_broken_module_is_skipped(tmp_path):
    (tmp_path / "ai-environments").mkdir()
    (tmp_path / "ai-environments" / "dace_ai_env_broken.py").write_text("this is not valid python !!!\n")
    with dace.config.set_temporary("default_build_folder", value=str(tmp_path)):
        # A hand-edited or truncated module must not make importing DaCe fail
        assert ai_environments.load_generated_environments() == []


@pytest.mark.parametrize("name", ["my-env", "my env", "2fft", "class", ""])
def test_names_must_be_identifiers(tmp_path, name):
    spec = EnvironmentSpec(name=name, headers=["x.h"])
    with dace.config.set_temporary("default_build_folder", value=str(tmp_path)):
        with pytest.raises(ai_environments.EnvironmentConflict, match="not a valid identifier"):
            ai_environments.materialize(spec)
    assert list(tmp_path.iterdir()) == []


def test_a_name_cannot_be_redefined(tmp_path):
    with dace.config.set_temporary("default_build_folder", value=str(tmp_path)):
        first = ai_environments.materialize(SPEC)
        # The same definition is reused
        assert ai_environments.materialize(SPEC.model_copy()).full_class_path() == first.full_class_path()

        for name in ("TestFFT", "testfft"):
            changed = SPEC.model_copy(update={"name": name, "cmake_libraries": ["fftw3f"]})
            with pytest.raises(ai_environments.EnvironmentConflict, match="different definition"):
                ai_environments.materialize(changed)

    # The existing definition is left alone
    assert "'fftw3f'" not in (tmp_path / "ai-environments" / "dace_ai_env_testfft.py").read_text()


def test_a_conflicting_environment_is_sent_back_to_the_model(tmp_path):
    sdfg = dace.SDFG("ai_env_conflict")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    node = AINode("needs_a_library", "Copy the value.", inputs={"_in"}, outputs={"_out"})
    state.add_node(node)
    state.add_edge(state.add_read("A"), None, node, "_in", dace.Memlet("A[0]"))
    state.add_edge(node, "_out", state.add_write("B"), None, dace.Memlet("B[0]"))

    bad = TaskletSpec(code="_out = _in;", environments=[SPEC.model_copy(update={"name": "Test-FFT"})])
    good = TaskletSpec(code="_out = _in;", environments=[SPEC])
    with dace.config.set_temporary("default_build_folder", value=str(tmp_path)):
        with dace.config.set_temporary("ai", "compile_check", value=False):
            with stub_provider_sequence([bad, good]) as provider:
                node.expand(state, "ai")

    assert len(provider.calls) == 2
    assert "cannot be used" in provider.calls[1][-1]["content"]
    assert "not a valid identifier" in provider.calls[1][-1]["content"]
    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    assert any(path.endswith(".TestFFT") for path in tasklet.environments)


def test_environment_dir_is_not_created_by_reading(tmp_path):
    target = tmp_path / "does_not_exist_yet"
    with dace.config.set_temporary("default_build_folder", value=str(target)):
        assert ai_environments.environment_dir() == str(target / "ai-environments")
        assert not target.exists(), "reading the environment directory must not create it"
        ai_environments.load_generated_environments()
        assert not target.exists()


def test_requested_environment_reaches_the_generated_tasklet(tmp_path):
    sdfg = dace.SDFG("ai_env_flow")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    node = AINode("needs_a_library", "Copy the value.", inputs={"_in"}, outputs={"_out"})
    state.add_node(node)
    state.add_edge(state.add_read("A"), None, node, "_in", dace.Memlet("A[0]"))
    state.add_edge(node, "_out", state.add_write("B"), None, dace.Memlet("B[0]"))

    spec = TaskletSpec(code="_out = _in;", environments=[SPEC])
    with dace.config.set_temporary("default_build_folder", value=str(tmp_path)):
        with dace.config.set_temporary("ai", "compile_check", value=False):
            with stub_provider(spec):
                node.expand(state, "ai")

    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    # ExpandTransformation.apply stores full class paths, which the code generator resolves later
    assert any(path.endswith(".TestFFT") for path in tasklet.environments), tasklet.environments
    for path in tasklet.environments:
        assert dace.library.get_environment(path).cmake_libraries == ["fftw3"]


def test_an_existing_environment_can_be_named_instead_of_described(tmp_path):
    """Reusing an installed environment must not write anything to disk."""
    with dace.config.set_temporary("default_build_folder", value=str(tmp_path)):
        env = ai_environments.materialize(SPEC)
    path = env.full_class_path()

    other = tmp_path / "nothing_written_here"
    with dace.config.set_temporary("default_build_folder", value=str(other)):
        assert ai_environments.resolve_installed([path]) == [env]
        # Naming the same environment twice attaches it once
        spec = TaskletSpec(code="_out = _in;", use_environments=[path, path])
        assert ai_environments.collect(spec) == [env]
    assert not other.exists(), "reusing an installed environment wrote a module"


def test_an_unknown_environment_reference_is_reported():
    with pytest.raises(AIExpansionError, match="not registered") as info:
        ai_environments.resolve_installed(["dace.libraries.nowhere.NoSuchEnvironment"])

    # The message must name what was asked for, so a repair or a human knows what to correct
    assert "NoSuchEnvironment" in str(info.value)


def test_reused_environments_reach_the_tasklet(tmp_path):
    with dace.config.set_temporary("default_build_folder", value=str(tmp_path)):
        env = ai_environments.materialize(SPEC)

    sdfg = dace.SDFG("ai_env_reuse")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    node = AINode("reuses_a_library", "Copy the value.", inputs={"_in"}, outputs={"_out"})
    state.add_node(node)
    state.add_edge(state.add_read("A"), None, node, "_in", dace.Memlet("A[0]"))
    state.add_edge(node, "_out", state.add_write("B"), None, dace.Memlet("B[0]"))

    spec = TaskletSpec(code="_out = _in;", use_environments=[env.full_class_path()])
    with dace.config.set_temporary("ai", "compile_check", value=False):
        with stub_provider(spec):
            node.expand(state, "ai")

    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    assert list(tasklet.environments) == [env.full_class_path()]


def test_the_prompt_separates_checked_from_unchecked_environments(monkeypatch):
    class Installed:
        is_installed = staticmethod(lambda: True)

    class Missing:
        is_installed = staticmethod(lambda: False)

    class Unchecked:
        pass

    registry = {"x.Installed": Installed, "x.Missing": Missing, "x.Unchecked": Unchecked}
    monkeypatch.setattr(dace.library, "_DACE_REGISTERED_ENVIRONMENTS", registry)

    sdfg = dace.SDFG("ai_env_listing")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    node = AINode("copy", "Copy the value.", inputs={"_in"}, outputs={"_out"})
    state.add_edge(state.add_read("A"), None, node, "_in", dace.Memlet("A[0]"))
    state.add_edge(node, "_out", state.add_write("B"), None, dace.Memlet("B[0]"))
    prompt = build_user_prompt(collect_context(node, state, sdfg))

    checked, _, unchecked = prompt.partition("without an installation check")
    assert "x.Installed" in checked and "x.Unchecked" in unchecked
    assert "x.Missing" not in prompt


if __name__ == "__main__":
    with offline():
        test_render_produces_every_required_field()
        test_materialize_writes_and_registers(pathlib.Path(tempfile.mkdtemp()))
        test_generated_environments_reload(pathlib.Path(tempfile.mkdtemp()))
        test_broken_module_is_skipped(pathlib.Path(tempfile.mkdtemp()))
        for name in ["my-env", "my env", "2fft", "class", ""]:
            test_names_must_be_identifiers(pathlib.Path(tempfile.mkdtemp()), name)
        test_a_name_cannot_be_redefined(pathlib.Path(tempfile.mkdtemp()))
        test_a_conflicting_environment_is_sent_back_to_the_model(pathlib.Path(tempfile.mkdtemp()))
        test_environment_dir_is_not_created_by_reading(pathlib.Path(tempfile.mkdtemp()))
        test_requested_environment_reaches_the_generated_tasklet(pathlib.Path(tempfile.mkdtemp()))
        test_an_existing_environment_can_be_named_instead_of_described(pathlib.Path(tempfile.mkdtemp()))
        test_an_unknown_environment_reference_is_reported()
        test_reused_environments_reach_the_tasklet(pathlib.Path(tempfile.mkdtemp()))
