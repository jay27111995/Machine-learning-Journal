import numpy as np 
import matplotlib.pyplot as plt

x = np.array(range(0, 2))

a = 1 + (3 * x)
b = 6 - (4 * x)
c = 3 - (2 * x)
d = 5 - (5 * x)

plt.plot(x, a)
plt.plot(x, b)
plt.plot(x, c)
plt.plot(x, d)

plt.show()
