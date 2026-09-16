# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Unit tests for the root conftest.py GPU worker/rank pinning logic.
"""
import importlib.util
import os
import pathlib
import types

from dace.sdfg.sdfg import LAUNCHER_RANK_VARS

CONFTEST_PATH = pathlib.Path(__file__).resolve().parent.parent / 'conftest.py'


def load_root_conftest() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location('dace_root_conftest_under_test', CONFTEST_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


root_conftest = load_root_conftest()


def clear_worker_env(monkeypatch) -> None:
    monkeypatch.delenv('PYTEST_XDIST_WORKER', raising=False)
    for var in LAUNCHER_RANK_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)


def test_parse_worker_index_gw_prefix():
    assert root_conftest.parse_worker_index('gw7') == 7


def test_parse_worker_index_fallback_no_digits():
    assert root_conftest.parse_worker_index('master') == 0
    assert root_conftest.parse_worker_index('gw') == 0


def test_pick_gpu_worker_device_modulo_wrap():
    assert root_conftest.pick_gpu_worker_device('gw4', ['0', '1']) == '0'
    assert root_conftest.pick_gpu_worker_device('gw5', ['0', '1']) == '1'


def test_resolve_worker_id_prefers_xdist_over_mpi_rank(monkeypatch):
    clear_worker_env(monkeypatch)
    monkeypatch.setenv('PYTEST_XDIST_WORKER', 'gw1')
    monkeypatch.setenv('OMPI_COMM_WORLD_RANK', '5')
    assert root_conftest.resolve_worker_id() == 'gw1'


def test_resolve_worker_id_uses_first_set_rank_var_in_order(monkeypatch):
    clear_worker_env(monkeypatch)
    monkeypatch.setenv('PMI_RANK', '9')
    monkeypatch.setenv('SLURM_PROCID', '1')
    assert root_conftest.resolve_worker_id() == '9'


def test_pin_worker_to_gpu_uses_preset_pool_and_rank_var(monkeypatch):
    clear_worker_env(monkeypatch)
    monkeypatch.setenv('OMPI_COMM_WORLD_RANK', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    root_conftest.pin_worker_to_gpu()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'


def test_pin_worker_to_gpu_noop_without_worker_id(monkeypatch):
    clear_worker_env(monkeypatch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    root_conftest.pin_worker_to_gpu()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'


def fake_config(numprocesses) -> types.SimpleNamespace:
    return types.SimpleNamespace(option=types.SimpleNamespace(numprocesses=numprocesses))


def test_is_xdist_controller_only_in_the_spawning_process(monkeypatch):
    clear_worker_env(monkeypatch)
    assert root_conftest.is_xdist_controller(fake_config(32))
    assert not root_conftest.is_xdist_controller(fake_config(None))
    assert not root_conftest.is_xdist_controller(fake_config(0))
    monkeypatch.setenv('PYTEST_XDIST_WORKER', 'gw0')
    assert not root_conftest.is_xdist_controller(fake_config(32))


def test_pin_worker_to_gpu_controller_keeps_whole_pool_for_its_workers(monkeypatch):
    clear_worker_env(monkeypatch)
    monkeypatch.setenv('SLURM_PROCID', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    root_conftest.pin_worker_to_gpu(controller=True)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'


def test_pin_worker_to_gpu_worker_rotates_despite_a_rank_var(monkeypatch):
    clear_worker_env(monkeypatch)
    monkeypatch.setenv('SLURM_PROCID', '0')
    monkeypatch.setenv('PYTEST_XDIST_WORKER', 'gw5')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    root_conftest.pin_worker_to_gpu()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
