import numpy as np

A = np.matrix([[1, 0.3], [0.45, 1.2]])
U, s, V = np.linalg.svd(A)

# We will use our created matrix A for this.
# print the matrices U, s, V

print(U)
print(s)
print(V)

# Lets verify some properties of the SVD matrices.
# Verify calculation of A=USV
print(np.allclose(A, U * np.diag(s) * V))

# Verify orthonormal properties of U and V. (Peformed on U but the same applies for V).
#  1) Dot product between columns = 0
print(np.round([np.dot(U[:, i-1].A1,  U[:, i].A1) for i in range(1, len(U))]))


#  2) Columns are unit vectors (length = 1)
print(np.round(np.sum((U*U), 0)))


#  3) Multiplying by its transpose = identity matrix
print(np.allclose(U.T * U, np.identity(len(U))))
