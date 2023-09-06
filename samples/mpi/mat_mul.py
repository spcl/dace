import numpy as np

dim_1 = 200
dim_2 = 300

a = np.arange(dim_1 * dim_2).reshape(dim_1, dim_2)
b = np.arange(dim_1 * dim_2).reshape(dim_2, dim_1)

def matrix_mul(a, b):
  a_mat = np.array(a)
  b_mat = np.array(b)
  c_mat = np.zeros((a_mat.shape[0], b_mat.shape[1]))

  if a_mat.shape[1] != b_mat.shape[0]:
    raise ValueError("A, B matrix dimension mismatched!")

  # more or less like C stationary
  for i in range(a_mat.shape[0]):
    for j in range(b_mat.shape[1]):
      for k in range(a_mat.shape[1]):
        c_mat[i][j] += a_mat[i][k] * b_mat[k][j]

  return c_mat


# print(matrix_mul(a,b))
# print(np.matmul(a,b))

print("Result correctness:", np.allclose(matrix_mul(a,b), np.matmul(a,b)))