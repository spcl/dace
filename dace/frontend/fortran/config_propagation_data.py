import json
from pathlib import Path
from typing import List, Any, Dict, Optional, Generator, Iterable

from dace.frontend.fortran.ast_desugaring import ConstTypeInjection, ConstInstanceInjection, ConstInjection, SPEC


def serialize(x: ConstInjection) -> str:
    assert isinstance(x, (ConstTypeInjection, ConstInstanceInjection))
    d: Dict[str, Any] = {'type': type(x).__name__,
                         'scope': '.'.join(x.scope_spec) if x.scope_spec else None,
                         'root': '.'.join(x.type_spec if isinstance(x, ConstTypeInjection) else x.root_spec),
                         'component': '.'.join(x.component_spec),
                         'value': x.value}
    return json.dumps(d)


def deserialize(s: str) -> ConstInjection:
    d = json.loads(s)
    assert d['type'] in {'ConstTypeInjection', 'ConstInstanceInjection'}
    scope = tuple(d['scope'].split('.')) if d['scope'] else None
    root = tuple(d['root'].split('.'))
    component = tuple(d['component'].split('.')) if d['component'] else None
    value = d['value']
    return ConstTypeInjection(scope, root, component, value) \
        if d['type'] == 'ConstTypeInjection' \
        else ConstInstanceInjection(scope, root, component, value)


def deserialize_v2(s: str,
                   typ: SPEC,
                   scope: Optional[SPEC] = None) -> List[ConstTypeInjection]:
    cfg = [tuple(ln.strip().split('=')) for ln in s.split('\n') if ln.strip()]
    cfg = [(k.strip(), v.strip()) for k, v in cfg]
    injs = []
    for k, v in cfg:
        kparts = tuple(k.split('.')[1:])  # Drop the first part that represents the type, but is not specific.
        if v == 'T':
            v = 'true'
        elif v == 'F':
            v = 'false'
        injs.append(ConstTypeInjection(scope, typ, kparts, v))
    return injs


def find_all_config_injection_files(root: Path) -> Generator[Path, None, None]:
    if root.is_file():
        yield root
    else:
        for f in root.rglob('*.ti'):
            yield f


def find_all_config_injections(ti_files: Iterable[Path]) -> Generator[ConstTypeInjection, None, None]:
    for f in ti_files:
        for l in f.read_text().strip().splitlines():
            if not l.strip():
                continue
            yield deserialize(l.strip())


def ecrad_config_injection_list(root: str = 'dace/frontend/fortran/conf_files') -> List[ConstTypeInjection]:
    cfgs = [Path(root).joinpath(f).read_text() for f in [
        'config.ti', 'aerosol_optics.ti', 'cloud_optics.ti', 'gas_optics_lw.ti', 'gas_optics_sw.ti', 'pdf_sampler.ti',
        'aerosol.ti', 'cloud.ti', 'flux.ti', 'gas.ti', 'single_level.ti', 'thermodynamics.ti']]
    injs = [deserialize(l.strip()) for c in cfgs for l in c.splitlines() if l.strip()]
    return injs
