# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import os
import subprocess
import importlib.util
import shutil
import dace

REPO_URL = "https://github.com/spcl/TransformationsTest.git"
REPO_NAME = "external_transformations"
EXPECTED_FILE = "empty_transformation.py"
EXPECTED_CLASS = "EmptyTransformation"
ENV_KEY = "DACE_external_transformations_path"


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


def _get_base_path() -> str:
    return str(dace.__external_transformations_path__)


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


def test_add_repository_and_check_file():
    _test_add_repository_and_check_file(ENV_KEY, None)
    _test_remove_repository(ENV_KEY, None)


def test_add_repository_and_check_file_with_env_var():
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
    _test_add_repository_and_check_file(ENV_KEY, None)
    _test_import_empty_transformation(ENV_KEY, None)
    _test_remove_repository(ENV_KEY, None)


def test_import_empty_transformation_with_env_var():
    base_path = _get_base_path()
    _test_add_repository_and_check_file(ENV_KEY, str(base_path))
    _test_import_empty_transformation(ENV_KEY, str(base_path))
    _test_remove_repository(ENV_KEY, str(base_path))


def _test_remove_repository(env_key: str, env_value: str):
    base_path = _get_base_path()
    repo_path = base_path / REPO_NAME
    result = run_cli(env_key, env_value, "remove", REPO_NAME)
    assert result.returncode == 0, "Failed to remove repository"
    assert not repo_path.exists(), "Repository folder still exists after removal"


def test_remove_repository():
    _test_add_repository_and_check_file(ENV_KEY, None)
    _test_remove_repository(ENV_KEY, None)


def test_remove_repository_with_env_var():
    base_path = _get_base_path()
    _test_add_repository_and_check_file(ENV_KEY, str(base_path))
    _test_remove_repository(ENV_KEY, str(base_path))


# Calling an external process won't see the dace.config setting we have changed
# This test would require a more complex setup to change the config.yaml to
# impact config once DaCe is imported or to override using environment variables with
# DACE_external_repository_path when get/set is called.
def test_add_repository_and_check_file_non_default_path():
    base_path = _set_and_get_non_default_path()
    _test_add_repository_and_check_file(ENV_KEY, str(base_path))
    _test_remove_repository(ENV_KEY, str(base_path))
    _set_default_path()


def test_import_empty_transformation_non_default_path():
    base_path = _set_and_get_non_default_path()
    _test_add_repository_and_check_file(ENV_KEY, str(base_path))
    _test_import_empty_transformation(ENV_KEY, str(base_path))
    _test_remove_repository(ENV_KEY, str(base_path))
    _set_default_path()


def test_remove_repository_non_default_path():
    base_path = _set_and_get_non_default_path()
    _test_add_repository_and_check_file(ENV_KEY, str(base_path))
    _test_remove_repository(ENV_KEY, str(base_path))
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
