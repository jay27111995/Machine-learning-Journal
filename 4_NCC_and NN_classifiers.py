import numpy as np

c1 = np.array([[-1, 0],
               [0, -1],
               [-0.5, -0.5],
               [-1.5, -1.5],
               [-2, 0],
               [0, -2],
               [-1, -1.3]])

c2 = np.array([[1, 1],
               [1.3, 0.7],
               [0.7, 1.3],
               [2.5, 1],
               [0, 1]])

X = np.array([[0, 0],
              [1, 1],
              [-1, 0],
              [0.7, -0.2],
              [-0.2, 1.5]])

# Find the centroids
C = np.empty((0, 2))
C = np.append(C, np.array([np.mean(c1, axis=0)]), axis=0)
C = np.append(C, np.array([np.mean(c2, axis=0)]), axis=0)
C = np.round(C, 4)
print(C)

# find the distance from the centroid
dnc = np.empty((0, 2))
lncarr = np.array([])
for i in X:
    d1 = np.linalg.norm(C[0] - i)
    d2 = np.linalg.norm(C[1] - i)
    dnc = np.append(dnc, np.array([[d1, d2]]), axis=0)
    if (d1 < d2):
        lncarr = np.append(lncarr, 1)
    else:
        lncarr = np.append(lncarr, 2)

print(dnc)
print(lncarr)

# Nearest Neighbor
dnn = np.empty((0, 2))
lnnarr = np.array([])
for i in X:
    d1 = None
    for j in c1: # For class 1
        d = np.linalg.norm(i - j)
        if d1 == None:
            d1 = d
        else:
            if d1 > d:
                d1 = d

    d2 = None
    for k in c2: # For class 2
        d = np.linalg.norm(i - k)
        if d2 == None:
            d2 = d
        else:
            if d2 > d:
                d2 = d

    dnn = np.append(dnn, np.array([[d1, d2]]), axis=0)
    if (d1 < d2):
        nc = 1
    else:
        nc = 2
    lnnarr = np.append(lnnarr, nc)

print(dnn)
print(lnnarr)