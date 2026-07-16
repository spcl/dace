# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Reproducer: ``compiler.cpu.codegen_params.const_scalar_abi`` changes how the readable CPU code
    generator binds a READ-ONLY scalar -- ``const T& x`` (``by_ref``, the default and the legacy
    convention) vs ``const T x`` (``by_value``, a copy).

    The two are SEMANTICALLY IDENTICAL. Which one is faster is a backend artifact, not a rule, and
    the sign is not fixed (LLVM #121262 has the reference winning; Lemire measured by-reference 4.5x
    slower on an aliasing-bound loop) -- which is exactly why this is a flag to sweep rather than a
    hardcoded choice. See CODEGEN_STYLE_PERFORMANCE.md.

    The binding is emitted by ``cpp.emit_memlet_reference`` for a NESTED-SDFG argument -- the only
    caller that passes an authoritative ``is_write=False`` -- so the fixture below is a nested SDFG
    with a read-only scalar in-connector. A plain tasklet copy-in is a non-const local and shows
    nothing. The legacy generator ignores the flag entirely and always binds by reference.

    Run:  python samples/codegen/const_scalar_abi_demo.py
"""
import dace
from dace.config import set_temporary

N = dace.symbol('N')


def make_sdfg(name='const_scalar_abi_demo'):
    """ out[j] *= s, with `s` a READ-ONLY scalar argument of a (non-inlined) nested SDFG. """
    inner = dace.SDFG('inner')
    inner.add_scalar('sc', dace.float64)
    inner.add_array('io', [N], dace.float64)
    istate = inner.add_state('n')
    entry, exit_node = istate.add_map('im', dict(j='0:N'))
    tasklet = istate.add_tasklet('it', {'v', 's'}, {'w'}, 'w = v * s')
    istate.add_memlet_path(istate.add_read('io'), entry, tasklet, dst_conn='v', memlet=dace.Memlet('io[j]'))
    istate.add_memlet_path(istate.add_read('sc'), entry, tasklet, dst_conn='s', memlet=dace.Memlet('sc[0]'))
    istate.add_memlet_path(tasklet, exit_node, istate.add_write('io'), src_conn='w', memlet=dace.Memlet('io[j]'))

    sdfg = dace.SDFG(name)
    sdfg.add_array('o', [N], dace.float64)
    sdfg.add_transient('tmp', [1], dace.float64)
    init = sdfg.add_state('init')
    mk = init.add_tasklet('mk', {}, {'r'}, 'r = 3.0')
    init.add_edge(mk, 'r', init.add_access('tmp'), None, dace.Memlet('tmp[0]'))
    use = sdfg.add_state_after(init, 'use')
    nsdfg = use.add_nested_sdfg(inner, {'sc', 'io'}, {'io'}, symbol_mapping={'N': N})
    nsdfg.no_inline = True  # keep it a real function so its argument list is emitted
    use.add_edge(use.add_read('tmp'), None, nsdfg, 'sc', dace.Memlet('tmp[0]'))
    use.add_edge(use.add_read('o'), None, nsdfg, 'io', dace.Memlet('o[0:N]'))
    use.add_edge(nsdfg, 'io', use.add_write('o'), None, dace.Memlet('o[0:N]'))
    sdfg.validate()
    return sdfg


def emit(sdfg, implementation, abi):
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'const_scalar_abi', value=abi):
        return '\n'.join((obj.clean_code or obj.code) for obj in sdfg.generate_code() if obj.language == 'cpp')


def scalar_arg_signature(code):
    """ The nested-SDFG signature line carrying the `sc` parameter. """
    for line in code.splitlines():
        if 'inner' in line and 'sc' in line and 'const double' in line:
            return line.strip()
    return '<not found>'


if __name__ == '__main__':
    sdfg = make_sdfg()
    print('compiler.cpu.codegen_params.const_scalar_abi -- emitted nested-SDFG signature\n')
    for implementation, abi in (('legacy', 'by_ref'), ('legacy', 'by_value'), ('experimental_readable', 'by_ref'),
                                ('experimental_readable', 'by_value')):
        signature = scalar_arg_signature(emit(sdfg, implementation, abi))
        print(f'  implementation={implementation:22s} abi={abi:9s}\n    {signature}\n')
    print('Note: legacy is IDENTICAL for both abi values -- it ignores the flag and always binds by\n'
          'reference, so its output stays byte-identical. Only experimental_readable honours it.')
