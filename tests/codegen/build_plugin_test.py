# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``compiler.build_plugin``: ``SDFG.compile`` hands the SDFG to an installed package's ``dace.build_plugins`` entry
point instead of building it the DaCe way."""

import pathlib
import sys
import tempfile

import pytest

import dace
from dace.config import set_temporary

PLUGIN = """
CALLS = []


def build(sdfg, **compile_args):
    CALLS.append((sdfg.name, compile_args))
    return "built by the plugin"
"""


def installed_plugin(root: pathlib.Path) -> None:
    """A package ``fake_build_plugin`` on ``sys.path`` that registers ``fake`` in ``dace.build_plugins``."""
    (root / "fake_build_plugin.py").write_text(PLUGIN)
    info = root / "fake_build_plugin-0.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fake-build-plugin\nVersion: 0.0\n")
    (info / "entry_points.txt").write_text("[dace.build_plugins]\nfake = fake_build_plugin:build\n")
    sys.path.insert(0, str(root))


def empty_program() -> dace.SDFG:
    sdfg = dace.SDFG("build_plugin_program")
    sdfg.add_state()
    return sdfg


def test_a_named_build_plugin_builds_the_program_with_the_compile_arguments():
    with tempfile.TemporaryDirectory() as root:
        installed_plugin(pathlib.Path(root))
        try:
            with set_temporary("compiler", "build_plugin", value="fake"):
                built = empty_program().compile(validate=False, return_program_handle=False)
            import fake_build_plugin

            assert built == "built by the plugin"
            assert fake_build_plugin.CALLS == [
                ("build_plugin_program", {"output_file": None, "validate": False, "return_program_handle": False})
            ]
        finally:
            sys.path.remove(root)
            sys.modules.pop("fake_build_plugin", None)


def test_a_build_plugin_no_package_registers_is_refused_by_name():
    with set_temporary("compiler", "build_plugin", value="not_installed"):
        with pytest.raises(ValueError, match="'not_installed'.*dace.build_plugins"):
            empty_program().compile()


if __name__ == "__main__":
    test_a_named_build_plugin_builds_the_program_with_the_compile_arguments()
    test_a_build_plugin_no_package_registers_is_refused_by_name()
