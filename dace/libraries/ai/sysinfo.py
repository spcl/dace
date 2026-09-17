# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Best-effort discovery of the local machine's CPU and GPU capabilities.

The AI library node expansion feeds this information to the model so that it can emit code tuned
for the machine the SDFG is actually being compiled on (which ISA extensions may be used, which
GPU architecture is targeted). Every function here is best-effort: it never raises, and returns
``None`` or an empty mapping when the information cannot be determined.

.. note::
   :func:`cpu_description` deliberately does not reuse :func:`dace.codegen.compiler.host_isa_id`,
   which hashes the same ``/proc/cpuinfo`` fields. That hash is part of the build cache key and
   must not change.
"""

import functools
import platform
import shutil
import subprocess
from typing import Dict, List, Optional

from dace.config import Config

#: Maximum time (in seconds) to wait for an external architecture-detection tool.
_PROBE_TIMEOUT = 10


def _run(args: List[str]) -> Optional[str]:
    """
    Runs an external probe and returns its stripped standard output, or ``None`` on any failure.

    :param args: Command line to run. ``args[0]`` is looked up on ``PATH`` first.
    :return: The command's standard output, or ``None`` if it is unavailable or failed.
    """
    if shutil.which(args[0]) is None:
        return None
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=_PROBE_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


@functools.lru_cache(maxsize=1)
def cpu_description() -> Dict[str, str]:
    """
    Describes the host CPU.

    :return: A mapping with the key ``machine`` (always present) and, where available, ``model``
             (the marketing name of the CPU) and ``flags`` (a space-separated list of ISA
             extensions as reported by the kernel).
    """
    info: Dict[str, str] = {'machine': platform.machine()}
    try:
        with open('/proc/cpuinfo', 'r') as fp:
            for line in fp:
                # 'Features' is the aarch64 spelling of 'flags'
                if line.startswith('model name') and 'model' not in info:
                    info['model'] = line.split(':', 1)[1].strip()
                elif line.startswith(('flags', 'Features')) and 'flags' not in info:
                    info['flags'] = line.split(':', 1)[1].strip()
                if 'model' in info and 'flags' in info:
                    break
    except OSError:
        pass
    # platform.processor() is only a fallback: on Linux it usually just repeats the machine type
    processor = platform.processor()
    if 'model' not in info and processor and processor != info['machine']:
        info['model'] = processor
    return info


def cpu_has_feature(feature: str) -> bool:
    """
    Checks whether the host CPU reports a given ISA extension.

    :param feature: The kernel's name for the extension, e.g. ``'avx2'``.
    :return: True if the flag is reported, False if it is absent or unknown.
    """
    return feature in cpu_description().get('flags', '').split()


@functools.lru_cache(maxsize=1)
def gpu_architectures() -> Optional[str]:
    """
    Returns the GPU architecture code will be compiled for.

    The configured architecture (``compiler.cuda.cuda_arch`` / ``compiler.cuda.hip_arch``) takes
    precedence, since that is what nvcc or hipcc will actually target; only when it is unset is the
    local hardware probed.

    .. note::
       A configured value is returned even when there is no local GPU, so this is *not* evidence
       that a device is present -- use :func:`gpu_names` for that.

    :return: A comma-separated architecture string (e.g. ``'sm_90'`` or ``'gfx942'``), or ``None``
             if none is configured and none could be detected.
    """
    from dace.codegen import common  # Avoid a cyclic import through the code generator

    try:
        backend = common.get_gpu_backend()
    except RuntimeError:
        return None

    if backend == 'cuda':
        configured = Config.get('compiler', 'cuda', 'cuda_arch')
        if configured:
            return ', '.join(f'sm_{a.strip()}' for a in configured.split(',') if a.strip())
        out = _run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'])
        if out:
            caps = sorted({'sm_' + line.strip().replace('.', '') for line in out.splitlines() if line.strip()})
            return ', '.join(caps) or None
        return None

    configured = Config.get('compiler', 'cuda', 'hip_arch')
    if configured:
        return ', '.join(a.strip() for a in configured.split(',') if a.strip())
    out = _run(['rocm_agent_enumerator'])
    if out:
        archs = sorted({line.strip() for line in out.splitlines() if line.strip().startswith('gfx')})
        return ', '.join(archs) or None
    return None


@functools.lru_cache(maxsize=1)
def gpu_names() -> Optional[str]:
    """
    Returns the marketing names of the local GPUs, if they can be determined.

    :return: A comma-separated list of device names, or ``None``.
    """
    out = _run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
    if out:
        return ', '.join(sorted({line.strip() for line in out.splitlines() if line.strip()}))
    out = _run(['rocm-smi', '--showproductname', '--csv'])
    if out:
        return out.splitlines()[-1].strip()
    return None
