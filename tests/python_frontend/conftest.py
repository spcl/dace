import contextlib
import importlib.util
import pathlib
import sys
import tempfile
import uuid

import pytest


@pytest.fixture
def temp_python_module():

    @contextlib.contextmanager
    def _load(module_source: str, module_name_prefix: str = 'dace_temp_module'):
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = pathlib.Path(temp_dir) / 'temp_module.py'
            module_path.write_text(module_source)

            module_name = f'{module_name_prefix}_{uuid.uuid4().hex}'
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            try:
                yield module
            finally:
                sys.modules.pop(module_name, None)

    return _load
