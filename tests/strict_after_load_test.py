import dace
import os


@dace.program
def multiple_fusions(A: dace.float32[10, 20], B: dace.float32[10, 20]):
    for i, j in dace.map[0:10, 0:20]:
        B[i, j] = A[i, j] + 1


if __name__ == '__main__':
    sdfg = multiple_fusions.to_sdfg(strict=False)
    sdfg.save(os.path.join('_dotgraphs', 'before.sdfg'))
    sdfg = dace.SDFG.from_file(os.path.join('_dotgraphs', 'before.sdfg'))
    sdfg.apply_strict_transformations()
    sdfg.compile()
