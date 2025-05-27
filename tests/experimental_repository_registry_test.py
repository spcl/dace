import os
import subprocess
import sys
from pathlib import Path
import importlib.util
import shutil
import pytest


REPO_URL = "https://github.com/ThrudPrimrose/LayoutTransformations.git"
REPO_NAME = "LayoutTransformations"
EXPECTED_FILE = "empty_transformation.py"
EXPECTED_CLASS = "EmptyTransformation"
CLONE_PATH = "layout_transformations"


def run_cli(*args):
    """Run the CLI script with arguments (bash-style)."""
    script_path = Path(__file__).parent.parent / "experimental_transformation_registry.py"
    cmd = ["python3", str(script_path), *args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Command: {' '.join(cmd)}")
        print(f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return result

def _get_base_path():
    import dace
    from dace import config

    base_path_home_unevaluated_str = config.Config.get("experimental_transformations_path")
    home_str = str(Path.home())
    base_path_str = base_path_home_unevaluated_str.replace('$HOME', home_str)
    base_path = Path(base_path_str)
    if not base_path.is_absolute():
        dace_root = Path(dace.__file__).resolve().parent.parent
        base_path = dace_root / base_path
    return base_path

def _set_and_get_non_default_path():
    from dace import config
    config.Config.set("experimental_transformations_path", value="dace/transformation/experimental")

    base_path = _get_base_path()
    return base_path

def _set_default_path():
    from dace import config
    config.Config.set("experimental_transformations_path", value="$HOME/dace_transformations/experimental_transformations")

    base_path = _get_base_path()
    return base_path


def _test_add_repository_and_check_file():
    base_path = _get_base_path()

    repo_path = base_path / REPO_NAME

    # Cleanup if it exists from previous failed test
    if repo_path.exists():
        shutil.rmtree(repo_path)

    result = run_cli("add", REPO_URL, "--force", "--name", REPO_NAME)
    assert result.returncode == 0, "Failed to add repo"

    assert repo_path.exists(), f"Repository directory {repo_path} does not exist"
    found = list(repo_path.rglob(EXPECTED_FILE))
    assert found, f"{EXPECTED_FILE} not found in repository"

def test_add_repository_and_check_file():
    _test_add_repository_and_check_file()
    _test_remove_repository()

#def test_add_repository_and_check_file_non_default_path():
#    _set_and_get_non_default_path()
#    _test_add_repository_and_check_file()
#    _test_remove_repository()
#    _set_default_path()

def _test_import_empty_transformation():
    _test_add_repository_and_check_file()
    base_path = _get_base_path()

    import_path = base_path / REPO_NAME
    file_path = list(import_path.rglob(EXPECTED_FILE))
    assert file_path, f"{EXPECTED_FILE} not found"

    spec = importlib.util.spec_from_file_location(
        f"experimental_transformations.{REPO_NAME}.{EXPECTED_FILE[:-3]}",
        file_path[0]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, EXPECTED_CLASS), f"{EXPECTED_CLASS} not found in module"

def test_import_empty_transformation():
    _test_add_repository_and_check_file()
    _test_import_empty_transformation()
    _test_remove_repository()

#def test_import_empty_transformation_non_default_path():
#    _set_and_get_non_default_path()
#    _test_add_repository_and_check_file()
#    _test_import_empty_transformation()
#    _test_remove_repository()
#    _set_default_path()

def _test_remove_repository():
    base_path = _get_base_path()

    repo_path = base_path / REPO_NAME

    result = run_cli("remove", REPO_NAME)
    assert result.returncode == 0, "Failed to remove repository"
    assert not repo_path.exists(), "Repository folder still exists after removal"

def test_remove_repository():
    _test_add_repository_and_check_file()
    _test_remove_repository()

#def test_remove_repository_non_default_path():
#    _set_and_get_non_default_path()
#    _test_add_repository_and_check_file()
#    _test_remove_repository()
#    _set_default_path()