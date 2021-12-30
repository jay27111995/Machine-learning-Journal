import numpy as np

# Bayes probability based classification
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

m1 = np.mean(c1, axis=0)
m2 = np.mean(c2, axis=0)

darr = np.array([])
larr = np.array([])
for i in X:
    d1 = np.linalg.norm(i - m1)
    d2 = np.linalg.norm(i - m2)
    px1 = round(np.exp(-d1)/(np.exp(-d1) + np.exp(-d2)), 4)
    px2 = round(np.exp(-d2)/(np.exp(-d1) + np.exp(-d2)), 4)

    pc1 = 7/12 # num of samples/total samples
    pc2 = 5/12

    p1x = px1 * pc1
    p2x = px2 * pc2

    darr = np.append(darr, (p1x - p2x))
    if (p1x - p2x) > 0:
        larr = np.append(larr, 1)
    else:
        larr = np.append(larr, 2)

print(larr)
print(darr)



