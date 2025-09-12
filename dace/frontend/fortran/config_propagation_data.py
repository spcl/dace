import json
import sys
from pathlib import Path
from typing import List, Any, Dict, Optional, Generator, Iterable, Tuple

from dace.frontend.fortran.ast_desugaring import ConstTypeInjection, ConstInstanceInjection, ConstInjection, SPEC


def serialize(x: ConstInjection) -> str:
    assert isinstance(x, (ConstTypeInjection, ConstInstanceInjection))
    d: Dict[str, Any] = {
        'type': type(x).__name__,
        'scope': '.'.join(x.scope_spec) if x.scope_spec else None,
        'root': '.'.join(x.type_spec if isinstance(x, ConstTypeInjection) else x.root_spec),
        'component': '.'.join(x.component_spec),
        'value': x.value
    }
    return json.dumps(d)


def deserialize(s: str) -> ConstInjection:
    d = json.loads(s)
    assert d['type'] in {'ConstTypeInjection', 'ConstInstanceInjection'}
    scope = tuple(d['scope'].split('.')) if d['scope'] else None
    root = tuple(d['root'].split('.'))
    component = tuple(d['component'].split('.')) if d['component'] else tuple()
    value = d['value']
    return ConstTypeInjection(scope, root, component, value) \
        if d['type'] == 'ConstTypeInjection' \
        else ConstInstanceInjection(scope, root, component, value)


def find_all_config_injection_files(root: Path) -> Generator[Path, None, None]:
    if root.is_file():
        yield root
    else:
        for f in root.rglob('*.ti'):
            yield f


def find_all_config_injections(ti_files: Iterable[Path]) -> Generator[ConstInjection, None, None]:
    inj_map: Dict[Tuple[str, str], ConstInjection] = {}
    for f in ti_files:
        for l in f.read_text().strip().splitlines():
            if not l.strip():
                continue
            x = deserialize(l.strip())
            if len(x.component_spec) > 1:
                print(f"{x}/{x.component_spec} must have just one-level for now; moving on...", file=sys.stderr)
                continue
            root = '.'.join(x.type_spec if isinstance(x, ConstTypeInjection) else x.root_spec)
            comp = '.'.join(x.component_spec)
            key = (root, comp)
            if key in inj_map:
                assert inj_map[key].value == x.value, \
                    f"Inconsistent values in constant injections: {x} vs. {inj_map[key]}"
            else:
                inj_map[key] = x
                yield x


def ecrad_config_injection_list(root: str = 'dace/frontend/fortran/conf_files') -> List[ConstInjection]:
    ti_files = [
        Path(root).joinpath(f) for f in [
            'config.ti', 'aerosol_optics.ti', 'cloud_optics.ti', 'gas_optics_lw.ti', 'gas_optics_sw.ti',
            'pdf_sampler.ti', 'aerosol.ti', 'cloud.ti', 'flux.ti', 'gas.ti', 'single_level.ti', 'thermodynamics.ti'
        ]
    ]
    return list(find_all_config_injections(ti_files))
