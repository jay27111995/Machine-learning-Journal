from numpy import array
from numpy.linalg import eig
from numpy.linalg import inv
from numpy import diag

# Create a matrix
A = array([[2, 3, 4], [4, 3, 2], [2, 5, 6]])
print(A)
# Eigen decomposition
values, vectors = eig(A)
print(values)
print(vectors)

# Verify Eigen decomposition
B = A.dot(vectors[:,0])
print(B)
C = vectors[:,0] * values[0]
print(C)

# Reconstruct the matrix
Q = vectors
# Create inverse matrix
Qinv = inv(Q)
# Create matrix with eigen values on diagonals
L = diag(values)
# Create original matrix from eigen values
D = Q.dot(L).dot(Qinv)
print(D)



