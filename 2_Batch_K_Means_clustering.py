import numpy as np

# Batch K-means clustering with 3 iterations
X = np.array([[-1, 0],
              [0, -1],
              [-0.5, -0.5],
              [-1.5, -1.5],
              [-2, 0],
              [0, -2],
              [-1, -1.3],
              [1, 1],
              [1.3, 0.7],
              [0.7, 1.3],
              [2.5, 1],
              [0, 1]])

M = np.array([[-1, -1], [-0.9, 0]])

for i in range(0, 3):
    darr = np.empty((0, 2))
    larr = np.array([])
    l1arr = np.empty((0, 2))
    l2arr = np.empty((0, 2))
    for i in X:
        d1 = np.linalg.norm(M[0] - i)
        d2 = np.linalg.norm(M[1] - i)
        darr = np.append(darr, np.array([[d1, d2]]), axis=0)
        if (d1 < d2):
            larr = np.append(larr, 1)
            l1arr = np.append(l1arr, np.array([i]), axis=0)
        else:
            larr = np.append(larr, 2)
            l2arr = np.append(l2arr, np.array([i]), axis=0)

    # Re-calculate the mean
    m1 = np.mean(l1arr, axis=0)
    m2 = np.mean(l2arr, axis=0)

    M = np.empty((0, 2))
    M = np.append(M, np.array([m1]), axis=0)
    M = np.append(M, np.array([m2]), axis=0)

    print((20 * "*") + "Iteration" + (20 * "*"))
    print(darr)
    print(larr)
    print(l1arr, l2arr)
    print(m1, m2)
    print(M)

print(50 * "-")
print(larr)