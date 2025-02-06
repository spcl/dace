import json
from typing import List, Any, Dict, Optional

from dace.frontend.fortran.ast_desugaring import ConstTypeInjection, ConstInstanceInjection, ConstInjection, SPEC


def serialize(x: ConstInjection) -> str:
    assert isinstance(x, (ConstTypeInjection, ConstInstanceInjection))
    d: Dict[str, Any] = {'type': type(x).__name__,
                         'scope': '.'.join(x.scope_spec) if x.scope_spec else None,
                         'root': '.'.join(x.type_spec if isinstance(x, ConstTypeInjection) else x.root_spec),
                         'component': '.'.join(x.component_spec) if x.component_spec else None,
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
