# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import os
import pathlib
import subprocess
import importlib.util
import shutil
import dace
import json
import tempfile

REPO_URL = "https://github.com/spcl/TransformationsTest.git"
REPO_NAME = "external_transformations"
EXPECTED_FILE = "empty_transformation.py"
EXPECTED_CLASS = "EmptyTransformation"
ENV_KEY = "DACE_external_transformations_path"
JSON_SAMPLE = """{
    "external_transformations": {
        "url": "https://github.com/spcl/TransformationsTest.git"
    },
    "renamed_external_transformations": {
        "url": "https://github.com/spcl/TransformationsTest.git"
    }
}
"""


def run_cli(env_key: str, env_value: str, *args):
    """Run the CLI script with a single environment variable."""
    cmd = ["dace-external-transformation-registry", *args]

    env = os.environ.copy()
    if env_key is not None and env_value is not None:
        env[env_key] = env_value
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elif env_key is not None and env_value is None:
        if env_key in os.environ:
            os.environ.pop(env_key, None)
            del env[env_key]  # Ensure the env variable is not set
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"Command: {' '.join(cmd)}")
        if env_key is not None and env_value is not None:
            print(f"Environment: {env_key}={env_value}")
        print(f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return result


def _get_base_path() -> pathlib.Path:
    return pathlib.Path(dace.__external_transformations_path__)


def _set_and_get_non_default_path() -> str:
    dace.config.Config.set("external_transformations_path", value="dace/transformation/external")
    base_path = _get_base_path()
    return base_path


def _set_default_path() -> str:
    dace.config.Config.set("external_transformations_path", value="$HOME/dace_transformations/external_transformations")
    base_path = _get_base_path()
    return base_path


def _test_add_repository_and_check_file(env_key: str, env_value: str):
    base_path = _get_base_path()

    repo_path = base_path / REPO_NAME

    # Cleanup if it exists from previous failed test
    if repo_path.exists():
        shutil.rmtree(repo_path)

    result = run_cli(env_key, env_value, "add", REPO_URL, "--force", "--name", REPO_NAME)
    assert result.returncode == 0, "Failed to add repo"

    assert repo_path.exists(), f"Repository directory {repo_path} does not exist"
    found = list(repo_path.rglob(EXPECTED_FILE))
    assert found, f"{EXPECTED_FILE} not found in repository"


def _test_load_from_file(env_key: str, env_value: str, filepath: str):
    base_path = _get_base_path()

    result = run_cli(env_key, env_value, "load-from-file", filepath, "--force")
    assert result.returncode == 0, "Failed to load repos from file"

    repo_path = base_path / REPO_NAME
    assert repo_path.exists(), f"Repository directory {repo_path} does not exist"
    repo_path = base_path / ("renamed_" + REPO_NAME)
    assert repo_path.exists(), f"Repository directory {repo_path} does not exist"
    found = list(repo_path.rglob(EXPECTED_FILE))
    assert found, f"{EXPECTED_FILE} not found in repository"


def test_add_repository_and_check_file():
    """
    Test adding a repository and checking that the expected file exists in the cloned repository. Cleans-up after test, which is also checked.
    """
    _test_add_repository_and_check_file(ENV_KEY, None)
    _test_remove_repository(ENV_KEY, None)


def test_add_repository_and_check_file_with_env_var():
    """
    Test adding a repository and checking the expected file exists, using an environment variable for the path.
    """
    base_path = _get_base_path()
    _test_add_repository_and_check_file(ENV_KEY, str(base_path))
    _test_remove_repository(ENV_KEY, str(base_path))


def _test_import_empty_transformation(env_key: str, env_value: str):

    _test_add_repository_and_check_file(env_key, env_value)
    base_path = _get_base_path()

    import_path = base_path / REPO_NAME
    file_path = list(import_path.rglob(EXPECTED_FILE))
    assert file_path, f"{EXPECTED_FILE} not found"

    spec = importlib.util.spec_from_file_location(f"external_transformations.{REPO_NAME}.{EXPECTED_FILE[:-3]}",
                                                  file_path[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, EXPECTED_CLASS), f"{EXPECTED_CLASS} not found in module"


def test_import_empty_transformation():
    """
    Test importing the expected transformation class from the cloned repository.
    """
    _test_add_repository_and_check_file(ENV_KEY, None)
    _test_import_empty_transformation(ENV_KEY, None)
    _test_remove_repository(ENV_KEY, None)


def test_import_empty_transformation_with_env_var():
    """
    Test importing the expected transformation class from the cloned repository using an environment variable for the path root path of external transformations.
    """
    base_path = _get_base_path()
    _test_add_repository_and_check_file(ENV_KEY, str(base_path))
    _test_import_empty_transformation(ENV_KEY, str(base_path))
    _test_remove_repository(ENV_KEY, str(base_path))


def _test_remove_repository(env_key: str, env_value: str, repo_name: str = REPO_NAME):
    base_path = _get_base_path()
    repo_path = base_path / repo_name
    result = run_cli(env_key, env_value, "remove", repo_name)
    assert result.returncode == 0, "Failed to remove repository"
    assert not repo_path.exists(), "Repository folder still exists after removal"


def test_remove_repository():
    """
    Test removing a repository and ensuring its directory is deleted.
    """
    _test_add_repository_and_check_file(ENV_KEY, None)
    _test_remove_repository(ENV_KEY, None)


def test_remove_repository_with_env_var():
    """
    Test removing a repository using an environment variable for the path.
    """
    base_path = _get_base_path()
    _test_add_repository_and_check_file(ENV_KEY, str(base_path))
    _test_remove_repository(ENV_KEY, str(base_path))


# Calling an external process won't see the dace.config setting we have changed
# This test would require a more complex setup to change the config.yaml to
# impact config once DaCe is imported or to override using environment variables with
# DACE_external_repository_path when get/set is called.
def test_add_repository_and_check_file_non_default_path():
    """
    Test adding a repository and checking the expected file exists using a non-default path.
    """
    base_path = _set_and_get_non_default_path()
    _test_add_repository_and_check_file(ENV_KEY, str(base_path))
    _test_remove_repository(ENV_KEY, str(base_path))
    _set_default_path()


def test_import_empty_transformation_non_default_path():
    """
    Test importing the expected transformation class from the cloned repository using a non-default path set through environment variables.
    """
    base_path = _set_and_get_non_default_path()
    _test_add_repository_and_check_file(ENV_KEY, str(base_path))
    _test_import_empty_transformation(ENV_KEY, str(base_path))
    _test_remove_repository(ENV_KEY, str(base_path))
    _set_default_path()


def test_remove_repository_non_default_path():
    """
    Test removing a repository and ensuring its directory is deleted using a non-default path.
    """
    base_path = _set_and_get_non_default_path()
    _test_add_repository_and_check_file(ENV_KEY, str(base_path))
    _test_remove_repository(ENV_KEY, str(base_path))
    _set_default_path()


def test_load_from_json_base_path():
    """
    Test loading repositories from a JSON file using the base path.
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_file:
        data = json.loads(JSON_SAMPLE)  # Ensure valid JSON data
        json.dump(data, tmp_file)
        tmp_file_path = tmp_file.name
    base_path = _get_base_path()
    _test_load_from_file(ENV_KEY, str(base_path), tmp_file_path)
    _test_remove_repository(ENV_KEY, str(base_path), "external_transformations")
    _test_remove_repository(ENV_KEY, str(base_path), "renamed_external_transformations")
    _set_default_path()


def test_load_from_json_non_base_path():
    """
    Test loading repositories from a JSON file using a non-default path.
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_file:
        data = json.loads(JSON_SAMPLE)  # Ensure valid JSON data
        json.dump(data, tmp_file)
        tmp_file_path = tmp_file.name
    base_path = _set_and_get_non_default_path()
    _test_load_from_file(ENV_KEY, str(base_path), tmp_file_path)
    _test_remove_repository(ENV_KEY, str(base_path), "external_transformations")
    _test_remove_repository(ENV_KEY, str(base_path), "renamed_external_transformations")
    _set_default_path()


if __name__ == "__main__":
    test_add_repository_and_check_file()
    test_import_empty_transformation()
    test_remove_repository()
    test_add_repository_and_check_file_with_env_var()
    test_import_empty_transformation_with_env_var()
    test_remove_repository_with_env_var()
    test_add_repository_and_check_file_non_default_path()
    test_import_empty_transformation_non_default_path()
    test_remove_repository_non_default_path()
    test_load_from_json_base_path()
    test_load_from_json_non_base_path()
