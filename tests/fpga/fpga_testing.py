#!/usr/bin/env python3
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import click
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
import re
import subprocess as sp
import sys
from typing import Union, Tuple

TEST_DIR = Path(__file__).absolute().parent.parent
DACE_DIR = TEST_DIR.parent


class Colors:
    SUCCESS = "\033[92m"
    STATUS = "\033[94m"
    ERROR = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_status(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    click.echo(
        f"{Colors.STATUS}{Colors.BOLD}[{timestamp}]{Colors.END} {message}")


def print_success(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    click.echo(
        f"{Colors.SUCCESS}{Colors.BOLD}[{timestamp}]{Colors.END} {message}")


def print_error(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    click.echo(
        f"{Colors.ERROR}{Colors.BOLD}[{timestamp}]{Colors.END} {message}")


def dump_logs(proc_or_logs: Union[sp.CompletedProcess, Tuple[str, str]]):
    if isinstance(proc_or_logs, tuple):
        log_out, log_err = proc_or_logs
    else:
        proc_or_logs.terminate()
        proc_or_logs.kill()
        try:
            log_out, log_err = proc_or_logs.communicate(timeout=10)
        except sp.TimeoutExpired:
            return None  # Failed to even kill the process
    if log_out:
        print(log_out)
    if log_err:
        print(log_err)
    return log_out, log_err


def run_parallel(test_func, tests, sequentialize):
    # Run tests in parallel using default number of workers
    with mp.Pool(1 if sequentialize else None) as pool:
        results = pool.starmap(test_func, tests)
        if all(results):
            print_success("All tests passed.")
            sys.exit(0)
        else:
            print_error("Failed tests:")
            for test, result in zip(tests, results):
                if result == False:
                    print_error(f"- {test[0]}")
            num_passed = sum(results, 0)
            num_tests = len(results)
            num_failed = num_tests - num_passed
            print_error(f"{num_passed} / {num_tests} tests passed "
                        f"({num_failed} tests failed).")
            sys.exit(1)


def cli(all_tests, test_func, tests_to_run, parallel):
    if tests_to_run:
        # If tests are specified on the command line, run only those tests, if
        # their name matches either the file or SDFG name of any known test
        test_dict = {t.replace(".py", ""): False for t in tests_to_run}
        test_patterns = {k: re.compile(k) for k in test_dict.keys()}
        to_run = []
        for t in all_tests:
            stem = Path(t[0]).stem
            sdfgs = t[1] if not isinstance(t[1], str) else [t[1]]
            for k, v in test_patterns.items():
                if re.search(v, stem):
                    to_run.append(t)
                    test_dict[k] = True
                    break
                for sdfg in sdfgs:
                    if re.search(v, sdfg):
                        to_run.append(t)
                        test_dict[k] = True
                        break
        for k, v in test_dict.items():
            if not v:
                raise ValueError(f"Test \"{k}\" not found.")
    else:
        # Otherwise run them all
        to_run = all_tests
    run_parallel(test_func, to_run, not parallel)
