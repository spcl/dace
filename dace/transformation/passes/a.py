f_cond = 1.0 if (a[i, j] > 0.5) else 0.0
c[i, j] = (a[i, j] * b[i, j]) * f_cond
          + (0.0) * (1.0 - f_cond)
d[i, j] = (1.0 - c[i, j]) * f_cond
          + (0.0) * (1.0 - f_cond)