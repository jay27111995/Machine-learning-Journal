import numpy as np

# There is some inconsistency between the results presented here and the text book
# Fuzzy K-means clustering with 3 iterations
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
Y = 2 # Fuzzification parameter

for i in range(0, 3):
    Marr = np.empty((0, 2))
    t1 = np.empty((1, 2))
    t2 = np.empty((1, 2))
    for i in X:
        d1 = np.linalg.norm(M[0] - i)
        d2 = np.linalg.norm(M[1] - i)
        a1 = np.power(d1, -Y)/(np.power(d1, -Y) + np.power(d2, -Y))
        a2 = np.power(d2, -Y)/(np.power(d1, -Y) + np.power(d2, -Y))
        Marr = np.append(Marr, np.array([[round(a1, 4), round(a2, 4)]]), axis=0)
        t1 = t1 + (a1 * i)
        t2 = t2 + (a2 * i)

    # Re-calculate the mean
    m1 = t1/((np.sum(Marr, axis=0))[0])
    m2 = t2/((np.sum(Marr, axis=0))[1])

    M = np.empty((0, 2))
    M = np.append(M, m1, axis=0)
    M = np.append(M, m2, axis=0)
    M = np.around(M, decimals=4)

    print((20 * "*") + "Iteration" + (20 * "*"))
    print(Marr)
    print(M)
