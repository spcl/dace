import dace

N = dace.symbol("N")

@dace.program
def branch_dependent_value_write(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    c: dace.float64[N, N],
    d: dace.float64[N, N],
):
    for i,j in dace.map[0:N:1, 0:N:1]:
        if a[i, j] > 0.5:
            c[i, j] = a[i, j] * b[i, j]
            d[i, j] = 1 - c[i, j]
        else:
            c[i, j] = 0.0
            d[i, j] = 0.0
        # Expression can be fused as:
        # mask[i, j] = double(a[i, j] > 1.0)
        # c[i, j] = a[i, j] * b[i, j] * mask[i,j] + (1 - mask[i,j]) * 0.0
        # d[i, j] = (1 - * c[i, j]) * mask[i,j] + (1 - mask[i,j]) * 0.0
        # Simplified to as:
        # c[i, j] = a[i, j] * b[i, j] * mask[i,j]
        # d[i, j] = (1 - * c[i, j]) * mask[i,j]


@dace.program
def branch_dependent_value_write_two(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    c: dace.float64[N, N],
    d: dace.float64[N, N],
):
    for i,j in dace.map[0:N:1, 0:N:1]:
        if a[i ,j] == 2.0:
            b[i, j] = 1.1
            d[i, j] = 0.8
        else:
            b[i, j] = -1.1
            d[i, j] = 2.2
        # Expression can be fused as:
        # mask[i, j] = double(a[i, j] == 2.0)
        # mask is either 1 or 0, so only one of the values are writetn to
        # c[i, j] = (1.1 * mask[i,j]) + ((1 - mask[i,j]) * -1.1)
        # d[i, j] = (0.8 * mask[i,j]) + ((1 - mask[i,j]) * 2.2)
        # can be simplified as (if detected): d[i, j] = (2.2 - 1.4 * mask[i,j])
        c[i, j] = max(0, b[i, j])
        d[i, j] = max(0, d[i, j])