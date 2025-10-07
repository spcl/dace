def spmv_inner(rowptr, colind, values, x, y):
    for i in range(N):
        sum_val = 0.0
        for j in range(rowptr[i], rowptr[i + 1]):
            sum_val += values[j] * x[colind[j]]
        y[i] = sum_val

def spmv_vectorized(rowptr, colind, values, x, VLEN, y):
    for i in range(N):
        sum_vec = [0.0] * VLEN
        for j in range(rowptr[i], rowptr[i + 1], VLEN):
            vals = values[j : j + VLEN]
            cols = colind[j : j + VLEN]
            x_vals = gather(x, cols)
            sum_vec = mul(x_vals, vals)
        y[i] = sum(sum_vec)