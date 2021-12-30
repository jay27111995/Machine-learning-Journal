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

l1 = np.array([[0, 1],
              [1, 0]])

l2 = np.array([[0.4, 0.8],
              [0.6, 0.2]])

m1 = np.mean(c1, axis=0)
m2 = np.mean(c2, axis=0)

pcxarr = np.empty((0,2))
for i in X:
    d1 = np.linalg.norm(i - m1)
    d2 = np.linalg.norm(i - m2)
    px1 = round(np.exp(-d1)/(np.exp(-d1) + np.exp(-d2)), 4)
    px2 = round(np.exp(-d2)/(np.exp(-d1) + np.exp(-d2)), 4)

    pc1 = 7/12 # num of samples/total samples
    pc2 = 5/12

    p1x = px1 * pc1
    p2x = px2 * pc2

    pcxarr = np.append(pcxarr, np.array([[p1x, p2x]]), axis=0)

print(pcxarr)
R1arr = np.dot(l1, np.transpose(pcxarr))
R2arr = np.dot(l2, np.transpose(pcxarr))
print(R1arr)
print(R2arr)

l1arr = np.array([])
l2arr = np.array([])

for i in range(0, len(R1arr[0])):
    if R1arr[0][i] < R1arr[1][i]:
        l1arr = np.append(l1arr, 1)
    else:
        l1arr = np.append(l1arr, 2)
    if R2arr[0][i] < R2arr[1][i]:
        l2arr = np.append(l2arr, 1)
    else:
        l2arr = np.append(l2arr, 2)

print(l1arr)
print(l2arr)
