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
    script_path = Path(__file__).parent.parent / "dace" / "transformation" / "experimental" / "registry.py"
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


def test_add_repository_and_check_file():
    repo_path = Path(__file__).parent.parent / "dace" / "transformation" / "experimental" / REPO_NAME

    # Cleanup if it exists from previous failed test
    if repo_path.exists():
        shutil.rmtree(repo_path)

    result = run_cli("add", REPO_URL, "--name", REPO_NAME)
    assert result.returncode == 0, "Failed to add repo"

    assert repo_path.exists(), f"Repository directory {repo_path} does not exist"
    found = list(repo_path.rglob(EXPECTED_FILE))
    assert found, f"{EXPECTED_FILE} not found in repository"


def test_import_empty_transformation():
    import_path = Path(__file__).parent.parent / "dace" / "transformation" / "experimental" / REPO_NAME
    file_path = list(import_path.rglob(EXPECTED_FILE))
    assert file_path, f"{EXPECTED_FILE} not found"

    spec = importlib.util.spec_from_file_location(
        f"dace.transformation.experimental.{REPO_NAME}.{EXPECTED_FILE[:-3]}",
        file_path[0]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, EXPECTED_CLASS), f"{EXPECTED_CLASS} not found in module"


def test_remove_repository():
    repo_path = Path(__file__).parent.parent / "dace" / "transformation" / "experimental" / REPO_NAME

    result = run_cli("remove", REPO_NAME)
    assert result.returncode == 0, "Failed to remove repository"
    assert not repo_path.exists(), "Repository folder still exists after removal"
