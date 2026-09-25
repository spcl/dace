# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A pointer alias of a read-only kernel argument keeps the argument's ``const``.

A kernel argument nothing writes is declared ``const T*`` in the launch signature, and the
dispatcher registers it that way. A View of such an array takes its ctype from the host
declaration, which has no qualifier, so an alias of it was emitted as ``T*`` and hipcc
rejected the initialization: xsbench's indirection into ``index_grid`` on the canon GPU column
failed with ``cannot initialize a variable of type 'int *__restrict' with an rvalue of type
'const int *'``.
"""
import dace

N, M, K = (dace.symbol(name) for name in ('N', 'M', 'K'))


@dace.program
def gather_through_a_view(grid: dace.int32[N, M], idx: dace.int32[K], out: dace.int32[K]):
    """``out = grid.flat[idx]``: the gather reads the view through a pointer connector."""
    flat = grid.reshape((N * M, ))
    out[:] = flat[idx]


def kernel_code() -> str:
    """The generated GPU code for the gather."""
    sdfg = gather_through_a_view.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    # Pin the block size: an unset one is only a default-configuration notice, and this test
    # treats warnings as errors.
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device:
            node.map.gpu_block_size = [128, 1, 1]
    return '\n'.join(part.clean_code for part in sdfg.generate_code())


def alias_lines(code: str) -> list[str]:
    """Every pointer-alias definition the tasklet emits."""
    return [line.strip() for line in code.splitlines() if '__arr = &' in line]


def test_the_alias_of_a_read_only_view_is_const():
    code = kernel_code()
    aliases = alias_lines(code)
    assert aliases, 'the indirection emitted no pointer alias'
    assert all(line.startswith('const ') for line in aliases), aliases


def test_the_argument_it_aliases_is_declared_const():
    """The signature and the alias have to agree; the argument side was already right."""
    code = kernel_code()
    signature = [line for line in code.splitlines() if '__global__' in line and 'gather' in line]
    assert signature, code[:400]
    assert 'const int * __restrict__ grid' in signature[0], signature[0]
